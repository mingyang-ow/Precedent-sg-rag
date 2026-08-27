from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr

from sg_legal_rag.api.app import create_app
from sg_legal_rag.api.provider import (
    MalformedGeneratedOutput,
    OpenAIProductionProvider,
    ProductionGenerationResult,
    ProviderExecutionError,
)
from sg_legal_rag.api.retrieval import RetrievalUnavailable
from sg_legal_rag.api.service import RAGApplicationService
from sg_legal_rag.api.settings import ApiSettings
from sg_legal_rag.generation.evidence import EvidenceItem, EvidenceOrigin, EvidencePackage
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_CITATION_CONTRACT_VERSION,
    ProductionAnswer,
    ProductionClaim,
)
from sg_legal_rag.generation.provider import TokenUsage
from sg_legal_rag.generation.schema import AnswerStatus


def evidence_item(
    evidence_id: str,
    *,
    case_id: str,
    passage: str,
    rank: int,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        case_id=case_id,
        case_name=f"Synthetic Authority {case_id}",
        source_judgment=f"[2023] SGHC {rank}",
        source_url=f"https://example.test/{case_id}",
        source_year=2023,
        passage=passage,
        passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        retrieval_rank=rank,
        retrieval_score=10.0 / rank,
        origin=EvidenceOrigin.HISTORICAL_RETRIEVAL,
        gold_row_id=None,
        citation_relationship_verified=True,
    )


EVIDENCE = (
    evidence_item(
        "E1",
        case_id="case:941",
        passage="The court identified three cumulative requirements.",
        rank=1,
    ),
    evidence_item(
        "E2",
        case_id="case:942",
        passage="The authority also emphasized the need for medical evidence.",
        rank=2,
    ),
)


@dataclass
class FakeRetriever:
    ready: bool = True
    evidence: tuple[EvidenceItem, ...] = EVIDENCE
    calls: list[tuple[str, int]] = field(default_factory=list)

    def is_ready(self) -> bool:
        return self.ready

    def retrieve(self, query_text: str, *, top_k: int) -> tuple[EvidenceItem, ...]:
        if not self.ready:
            raise RetrievalUnavailable("synthetic retrieval failure")
        self.calls.append((query_text, top_k))
        return self.evidence[:top_k]


@dataclass
class FakeProvider:
    behavior: str = "answer"
    calls: list[EvidencePackage] = field(default_factory=list)
    provider_name: str = field(default="fake", init=False)
    model_family: str = field(default="fake", init=False)

    def generate(self, package: EvidencePackage) -> ProductionGenerationResult:
        self.calls.append(package)
        if self.behavior == "failure":
            raise ProviderExecutionError("synthetic upstream failure")
        if self.behavior == "malformed":
            raise MalformedGeneratedOutput("synthetic malformed output")
        if self.behavior == "unexpected":
            raise RuntimeError("sk-private-upstream-exception")
        if self.behavior == "abstain":
            answer = ProductionAnswer(
                contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                recommended_case_id=None,
                explanation="The supplied evidence does not support a precedent recommendation.",
                claims=(),
            )
        elif self.behavior == "invalid_evidence":
            answer = _answered(evidence_id="E99")
        elif self.behavior == "case_mismatch":
            answer = _answered(case_id="case:942")
        elif self.behavior == "no_claims":
            answer = ProductionAnswer(
                contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
                status=AnswerStatus.ANSWERED,
                recommended_case_id="case:941",
                explanation="A structurally incomplete generated answer.",
                claims=(),
            )
        else:
            answer = _answered()
        return ProductionGenerationResult(
            answer=answer,
            provider_status="fake_succeeded",
            latency_ms=0.1,
            usage=TokenUsage(
                input_tokens=100,
                cached_input_tokens=20,
                output_tokens=30,
                reasoning_tokens=5,
                total_tokens=130,
            ),
            estimated_cost_usd=0.00005,
        )


def _answered(*, evidence_id: str = "E1", case_id: str = "case:941") -> ProductionAnswer:
    return ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.ANSWERED,
        recommended_case_id="case:941",
        explanation="The authority states the test, but factual application remains unresolved.",
        claims=(
            ProductionClaim(
                statement="The authority identifies three cumulative requirements.",
                evidence_id=evidence_id,
                case_id=case_id,
            ),
        ),
    )


def settings(*, api_key: str | None = None) -> ApiSettings:
    return ApiSettings(
        model_id="test-model",
        top_k_default=2,
        max_top_k=3,
        openai_api_key=SecretStr(api_key) if api_key else None,
    )


def client_for(
    *,
    retriever: FakeRetriever | None = None,
    provider: FakeProvider | None = None,
    api_key: str | None = None,
) -> tuple[OfflineASGIClient, FakeRetriever, FakeProvider | None]:
    retrieval = retriever or FakeRetriever()
    service = RAGApplicationService(
        settings=settings(api_key=api_key),
        retriever=retrieval,
        provider=provider,
    )
    return (
        OfflineASGIClient(create_app(settings=service.settings, service=service)),
        retrieval,
        provider,
    )


class OfflineASGIClient:
    """Exercise the ASGI app directly without Starlette's synchronous portal."""

    def __init__(self, application: Any) -> None:
        self.application = application

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            async with (
                self.application.router.lifespan_context(self.application),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=self.application),
                    base_url="http://testserver",
                ) as client,
            ):
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)


def metric_value(text: str, name: str, **labels: str) -> float:
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                return sample.value
    raise AssertionError(f"metric sample not found: {name} {labels}")


def test_health_is_cheap_and_returns_200() -> None:
    client, retriever, _ = client_for()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert retriever.calls == []
    assert response.headers["X-Request-ID"]


def test_metrics_endpoint_is_passive_prometheus_output_without_sensitive_labels() -> None:
    provider = FakeProvider()
    client, retriever, _ = client_for(provider=provider)

    response = client.get("/metrics", headers={"X-Request-ID": "private-request-id"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert retriever.calls == []
    assert provider.calls == []
    assert "precedent_http_requests_total" in response.text
    assert "precedent_retrieval_ready 1.0" in response.text
    for prohibited in (
        "request_id=",
        "query=",
        "evidence_id=",
        "package_id=",
        "case_name=",
        "private-request-id",
    ):
        assert prohibited not in response.text


def test_http_metrics_use_route_templates_and_isolated_registries() -> None:
    first, _, _ = client_for()
    second, _, _ = client_for()

    assert first.get("/health").status_code == 200
    assert first.get("/not-a-real-private-path").status_code == 404
    first_metrics = first.get("/metrics").text
    second_metrics = second.get("/metrics").text

    assert (
        metric_value(
            first_metrics,
            "precedent_http_requests_total",
            method="GET",
            endpoint="/health",
            status_class="2xx",
        )
        == 1
    )
    assert (
        metric_value(
            first_metrics,
            "precedent_http_requests_total",
            method="GET",
            endpoint="unmatched",
            status_class="4xx",
        )
        == 1
    )
    assert "/not-a-real-private-path" not in first_metrics
    assert 'endpoint="/health"' not in second_metrics
    assert metric_value(second_metrics, "precedent_requests_in_flight") == 1


def test_readiness_is_partial_without_generation_and_makes_no_provider_call() -> None:
    client, retriever, _ = client_for(provider=None)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "partial",
        "retrieval": True,
        "generation_configured": False,
        "answer_ready": False,
    }
    assert retriever.calls == []


def test_readiness_returns_503_when_retrieval_assets_are_unavailable() -> None:
    client, _, _ = client_for(retriever=FakeRetriever(ready=False))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["retrieval"] is False


def test_retrieve_facts_only_uses_default_top_k_and_returns_stored_evidence() -> None:
    client, retriever, _ = client_for()

    response = client.post("/retrieve", json={"facts": "  diminished   responsibility  "})

    assert response.status_code == 200
    body = response.json()
    assert retriever.calls == [("diminished responsibility", 2)]
    assert len(body["results"]) == 2
    assert body["results"][0]["passage"] == EVIDENCE[0].passage
    assert body["results"][0]["passage_digest"] == EVIDENCE[0].passage_digest
    assert body["results"][0]["package_id"] == body["package_id"]
    assert body["timings"]["retrieval_ms"] >= 0


def test_retrieve_combines_facts_and_optional_principle() -> None:
    client, retriever, _ = client_for()

    response = client.post(
        "/retrieve",
        json={"facts": "criminal appeal", "principle": "medical evidence", "top_k": 1},
    )

    assert response.status_code == 200
    assert retriever.calls == [("criminal appeal medical evidence", 1)]
    assert len(response.json()["results"]) == 1


@pytest.mark.parametrize("top_k", [0, 4])
def test_retrieve_rejects_top_k_outside_configured_bounds(top_k: int) -> None:
    client, retriever, _ = client_for()

    response = client.post("/retrieve", json={"facts": "query", "top_k": top_k})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    assert retriever.calls == []
    metrics = client.get("/metrics").text
    assert (
        metric_value(
            metrics,
            "precedent_failures_total",
            category="request_validation_failure",
        )
        == 1
    )


def test_request_schema_rejects_blank_facts_and_unknown_fields() -> None:
    client, _, _ = client_for()

    blank = client.post("/retrieve", json={"facts": "   "})
    unknown = client.post("/retrieve", json={"facts": "query", "reranker": "hidden-knob"})

    assert blank.status_code == 422
    assert unknown.status_code == 422
    assert blank.json()["error"]["code"] == "request_validation_failed"
    metrics = client.get("/metrics").text
    assert (
        metric_value(
            metrics,
            "precedent_failures_total",
            category="request_validation_failure",
        )
        == 2
    )


def test_answer_returns_application_controlled_citation() -> None:
    provider = FakeProvider()
    client, _, _ = client_for(provider=provider)

    response = client.post("/answer", json={"facts": "diminished responsibility", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    citation = body["claims"][0]["citation"]
    assert body["status"] == "answered"
    assert body["contract_version"] == "production-citation-v1"
    assert citation["source_text"] == EVIDENCE[0].passage
    assert citation["passage_digest"] == EVIDENCE[0].passage_digest
    assert body["package_id"]
    assert citation["evidence_id"] == "E1"
    assert "supporting_quote" not in body["claims"][0]
    assert len(provider.calls) == 1


def test_answer_can_return_citation_free_abstention() -> None:
    client, _, _ = client_for(provider=FakeProvider(behavior="abstain"))

    response = client.post("/answer", json={"facts": "unrelated query"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["recommended_case_id"] is None
    assert response.json()["claims"] == []


def test_rag_answer_usage_cost_and_phase_metrics_update_offline() -> None:
    provider = FakeProvider()
    client, _, _ = client_for(provider=provider)

    assert client.post("/retrieve", json={"facts": "private facts"}).status_code == 200
    assert client.post("/answer", json={"facts": "private facts"}).status_code == 200
    provider.behavior = "abstain"
    assert client.post("/answer", json={"facts": "private facts"}).status_code == 200
    metrics = client.get("/metrics").text

    assert (
        metric_value(
            metrics,
            "precedent_rag_operations_total",
            operation="retrieve",
            outcome="success",
        )
        == 1
    )
    assert (
        metric_value(
            metrics,
            "precedent_rag_operations_total",
            operation="answer",
            outcome="success",
        )
        == 2
    )
    assert (
        metric_value(
            metrics,
            "precedent_answer_outcomes_total",
            status="answered",
        )
        == 1
    )
    assert (
        metric_value(
            metrics,
            "precedent_answer_outcomes_total",
            status="insufficient_evidence",
        )
        == 1
    )
    assert metric_value(metrics, "precedent_retrieval_duration_seconds_count") == 3
    assert metric_value(metrics, "precedent_generation_duration_seconds_count") == 2
    assert metric_value(metrics, "precedent_resolution_duration_seconds_count") == 2
    assert metric_value(metrics, "precedent_retrieval_results_count") == 3
    provider_labels = {"provider": "fake", "model_family": "fake"}
    assert metric_value(metrics, "precedent_provider_requests_total", **provider_labels) == 2
    assert (
        metric_value(metrics, "precedent_provider_duration_seconds_count", **provider_labels) == 2
    )
    assert metric_value(metrics, "precedent_llm_input_tokens_total", **provider_labels) == 200
    assert metric_value(metrics, "precedent_llm_cached_input_tokens_total", **provider_labels) == 40
    assert metric_value(metrics, "precedent_llm_output_tokens_total", **provider_labels) == 60
    assert metric_value(metrics, "precedent_llm_reasoning_tokens_total", **provider_labels) == 10
    assert metric_value(
        metrics,
        "precedent_llm_estimated_cost_usd_total",
        **provider_labels,
    ) == pytest.approx(0.0001)
    assert "private facts" not in metrics


def test_answer_without_generation_configuration_returns_503_before_retrieval() -> None:
    client, retriever, _ = client_for(provider=None)

    response = client.post("/answer", json={"facts": "query"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "generation_unavailable"
    assert retriever.calls == []
    metrics = client.get("/metrics").text
    assert (
        metric_value(
            metrics,
            "precedent_failures_total",
            category="generation_unavailable",
        )
        == 1
    )


@pytest.mark.parametrize(
    ("behavior", "expected_code"),
    [
        ("invalid_evidence", "invalid_generated_citation"),
        ("case_mismatch", "invalid_generated_citation"),
        ("no_claims", "invalid_generated_citation"),
        ("failure", "provider_failure"),
        ("malformed", "malformed_generated_output"),
        ("unexpected", "provider_failure"),
    ],
)
def test_answer_maps_generated_and_provider_failures(behavior: str, expected_code: str) -> None:
    client, _, _ = client_for(provider=FakeProvider(behavior=behavior))

    response = client.post("/answer", json={"facts": "query"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    metrics = client.get("/metrics").text
    category = (
        behavior
        if behavior in {"failure", "malformed", "unexpected"}
        else "citation_contract_failure"
    )
    category = {
        "failure": "provider_failure",
        "malformed": "malformed_generated_output",
        "unexpected": "provider_failure",
    }.get(category, category)
    assert metric_value(metrics, "precedent_failures_total", category=category) == 1
    if behavior in {"failure", "malformed", "unexpected"}:
        assert (
            metric_value(
                metrics,
                "precedent_provider_failures_total",
                provider="fake",
                model_family="fake",
                category=category,
            )
            == 1
        )
    else:
        issue_code = {
            "invalid_evidence": "unknown_evidence_id",
            "case_mismatch": "case_evidence_mismatch",
            "no_claims": "answer_without_supporting_evidence",
        }[behavior]
        assert (
            metric_value(
                metrics,
                "precedent_citation_contract_violations_total",
                issue_code=issue_code,
            )
            == 1
        )


def test_provider_exception_payload_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    client, _, _ = client_for(provider=FakeProvider(behavior="unexpected"))

    with caplog.at_level("INFO", logger="sg_legal_rag.api"):
        response = client.post("/answer", json={"facts": "private facts"})

    assert response.status_code == 502
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "sk-private-upstream-exception" not in rendered
    assert "private facts" not in rendered


def test_changed_evidence_digest_fails_as_internal_integrity_error() -> None:
    corrupt = EVIDENCE[0].model_copy(update={"passage_digest": "0" * 64})
    client, _, _ = client_for(
        retriever=FakeRetriever(evidence=(corrupt,)),
        provider=FakeProvider(),
    )

    response = client.post("/answer", json={"facts": "query", "top_k": 1})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "evidence_integrity_failure"
    assert response.json()["error"]["issues"] == ["evidence_digest_mismatch"]
    metrics = client.get("/metrics").text
    assert (
        metric_value(
            metrics,
            "precedent_failures_total",
            category="evidence_integrity_failure",
        )
        == 1
    )
    assert (
        metric_value(
            metrics,
            "precedent_citation_contract_violations_total",
            issue_code="evidence_digest_mismatch",
        )
        == 1
    )


def test_retrieval_unavailable_is_observable_without_exposing_failure_text() -> None:
    client, _, _ = client_for(retriever=FakeRetriever(ready=False))

    response = client.post("/retrieve", json={"facts": "sensitive unavailable query"})
    metrics = client.get("/metrics").text

    assert response.status_code == 503
    assert (
        metric_value(
            metrics,
            "precedent_failures_total",
            category="retrieval_unavailable",
        )
        == 1
    )
    assert (
        metric_value(
            metrics,
            "precedent_rag_operations_total",
            operation="retrieve",
            outcome="failure",
        )
        == 1
    )
    assert "sensitive unavailable query" not in metrics


def test_request_id_is_preserved_when_safe_and_replaced_when_invalid() -> None:
    client, _, _ = client_for()

    preserved = client.get("/health", headers={"X-Request-ID": "portfolio-demo_123"})
    replaced = client.get("/health", headers={"X-Request-ID": "unsafe id with spaces"})

    assert preserved.headers["X-Request-ID"] == "portfolio-demo_123"
    assert replaced.headers["X-Request-ID"] != "unsafe id with spaces"
    assert re.fullmatch(r"[0-9a-f-]{36}", replaced.headers["X-Request-ID"])


def test_logs_are_structured_and_never_contain_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-portfolio-secret"
    client, _, _ = client_for(provider=FakeProvider(), api_key=secret)

    with caplog.at_level("INFO", logger="sg_legal_rag.api"):
        response = client.post("/answer", json={"facts": "sensitive user facts", "top_k": 1})

    assert response.status_code == 200
    assert secret not in response.text
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in rendered
    assert "sensitive user facts" not in rendered
    events = [json.loads(record.getMessage()) for record in caplog.records]
    operation = next(event for event in events if event["event"] == "rag_operation")
    assert operation["retrieval_count"] == 1
    assert operation["answer_status"] == "answered"
    assert operation["provider_status"] == "fake_succeeded"
    assert "generation_ms" in operation


def test_health_and_readiness_do_not_construct_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    def forbidden_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("OpenAI client must not be constructed")

    monkeypatch.setattr(openai, "OpenAI", forbidden_client)
    configured = settings(api_key="sk-test-never-used")
    application = create_app(settings=configured)
    client = OfflineASGIClient(application)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code in {200, 503}


def test_openai_adapter_records_reported_usage_and_frozen_cost_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    calls: list[dict[str, Any]] = []

    class FakeResponses:
        def parse(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                output_parsed=_answered(),
                usage=SimpleNamespace(
                    input_tokens=100,
                    input_tokens_details=SimpleNamespace(cached_tokens=20),
                    output_tokens=30,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=5),
                    total_tokens=130,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["api_key"] == "sk-offline-test"
            self.responses = FakeResponses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    configured = ApiSettings(
        model_id="gpt-5.6-luna",
        top_k_default=2,
        max_top_k=3,
        openai_api_key=SecretStr("sk-offline-test"),
    )
    provider = OpenAIProductionProvider(
        api_key=configured.openai_api_key,
        model_id=configured.model_id,
        max_output_tokens=configured.max_output_tokens,
        reasoning_effort=configured.reasoning_effort,
        verbosity=configured.verbosity,
    )
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=provider,
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    response = client.post("/answer", json={"facts": "offline provider telemetry"})
    metrics = client.get("/metrics").text
    labels = {"provider": "openai", "model_family": "gpt-5.6-luna"}

    assert response.status_code == 200
    assert len(calls) == 1
    assert metric_value(metrics, "precedent_provider_requests_total", **labels) == 1
    assert metric_value(metrics, "precedent_llm_input_tokens_total", **labels) == 100
    assert metric_value(metrics, "precedent_llm_cached_input_tokens_total", **labels) == 20
    assert metric_value(metrics, "precedent_llm_output_tokens_total", **labels) == 30
    assert metric_value(metrics, "precedent_llm_reasoning_tokens_total", **labels) == 5
    assert metric_value(
        metrics,
        "precedent_llm_estimated_cost_usd_total",
        **labels,
    ) == pytest.approx(0.0000524)


def test_openapi_exposes_small_typed_surface_and_contract_metadata() -> None:
    client, _, _ = client_for()

    schema = client.get("/openapi.json").json()
    version = client.get("/version")

    assert {"/health", "/ready", "/metrics", "/version", "/retrieve", "/answer"} <= set(
        schema["paths"]
    )
    assert "RetrieveRequest" in schema["components"]["schemas"]
    assert "AnswerResponse" in schema["components"]["schemas"]
    assert version.json()["citation_contract"] == "production-citation-v1"
    assert len(version.json()["schema_signature"]) == 64
