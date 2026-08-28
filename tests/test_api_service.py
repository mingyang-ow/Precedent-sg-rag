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
    ProviderTimeoutError,
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
        if self.behavior == "timeout":
            raise ProviderTimeoutError("synthetic upstream timeout")
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


SERVICE_KEY = "precedent-test-service-key"
METRICS_KEY = "precedent-test-metrics-key"


def settings(*, api_key: str | None = None, **overrides: Any) -> ApiSettings:
    values: dict[str, Any] = {
        "model_id": "test-model",
        "top_k_default": 2,
        "max_top_k": 3,
        "openai_api_key": SecretStr(api_key) if api_key else None,
        "precedent_api_key": SecretStr(SERVICE_KEY),
        "metrics_api_key": SecretStr(METRICS_KEY),
    }
    values.update(overrides)
    return ApiSettings(**values)


def client_for(
    *,
    retriever: FakeRetriever | None = None,
    provider: FakeProvider | None = None,
    api_key: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> tuple[OfflineASGIClient, FakeRetriever, FakeProvider | None]:
    retrieval = retriever or FakeRetriever()
    service = RAGApplicationService(
        settings=settings(api_key=api_key, **(settings_overrides or {})),
        retriever=retrieval,
        provider=provider,
    )
    return (
        OfflineASGIClient(
            create_app(settings=service.settings, service=service),
            headers={
                "X-Precedent-API-Key": SERVICE_KEY,
                "X-Precedent-Metrics-Key": METRICS_KEY,
            },
        ),
        retrieval,
        provider,
    )


class OfflineASGIClient:
    """Exercise the ASGI app directly without Starlette's synchronous portal."""

    def __init__(self, application: Any, *, headers: dict[str, str] | None = None) -> None:
        self.application = application
        self.headers = headers or {}

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        request_headers = {**self.headers, **kwargs.pop("headers", {})}

        async def send() -> httpx.Response:
            async with (
                self.application.router.lifespan_context(self.application),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=self.application),
                    base_url="http://testserver",
                ) as client,
            ):
                return await client.request(method, path, headers=request_headers, **kwargs)

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
            assert kwargs["max_retries"] == 0
            assert kwargs["timeout"] == 17.0
            self.responses = FakeResponses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    configured = settings(api_key="sk-offline-test", model_id="gpt-5.6-luna")
    provider = OpenAIProductionProvider(
        api_key=configured.openai_api_key,
        model_id=configured.model_id,
        max_output_tokens=configured.max_output_tokens,
        reasoning_effort=configured.reasoning_effort,
        verbosity=configured.verbosity,
        timeout_seconds=17.0,
    )
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=provider,
    )
    client = OfflineASGIClient(
        create_app(settings=configured, service=service),
        headers={
            "X-Precedent-API-Key": SERVICE_KEY,
            "X-Precedent-Metrics-Key": METRICS_KEY,
        },
    )

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
    assert set(schema["components"]["securitySchemes"]) == {
        "PrecedentMetricsKey",
        "PrecedentServiceKey",
    }
    assert schema["paths"]["/retrieve"]["post"]["security"] == [{"PrecedentServiceKey": []}]
    assert schema["paths"]["/answer"]["post"]["security"] == [{"PrecedentServiceKey": []}]
    assert schema["paths"]["/metrics"]["get"]["security"] == [{"PrecedentMetricsKey": []}]
    assert "security" not in schema["paths"]["/health"]["get"]
    assert version.json()["citation_contract"] == "production-citation-v1"
    assert len(version.json()["schema_signature"]) == 64


def test_auth_policy_keeps_status_routes_public_and_protects_business_routes() -> None:
    configured = settings()
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=FakeProvider(),
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    for path in ("/health", "/ready", "/version", "/docs", "/openapi.json"):
        assert client.get(path).status_code == 200
    assert client.post("/retrieve", json={"facts": "query"}).status_code == 401
    assert client.post("/answer", json={"facts": "query"}).status_code == 401
    assert client.get("/metrics").status_code == 401


def test_service_and_metrics_credentials_are_separate() -> None:
    configured = settings()
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=FakeProvider(),
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    wrong = client.post(
        "/retrieve",
        json={"facts": "query"},
        headers={"X-Precedent-API-Key": METRICS_KEY},
    )
    metrics_with_service_key = client.get(
        "/metrics", headers={"X-Precedent-Metrics-Key": SERVICE_KEY}
    )
    authorized = client.post(
        "/retrieve",
        json={"facts": "query"},
        headers={"X-Precedent-API-Key": SERVICE_KEY},
    )
    metrics = client.get("/metrics", headers={"X-Precedent-Metrics-Key": METRICS_KEY})

    assert wrong.status_code == 401
    assert wrong.headers["WWW-Authenticate"] == "APIKey"
    assert metrics_with_service_key.status_code == 401
    assert authorized.status_code == 200
    assert metrics.status_code == 200
    assert (
        metric_value(metrics.text, "precedent_failures_total", category="authentication_failure")
        == 2
    )


def test_authentication_uses_constant_time_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sg_legal_rag.api.security as security_module

    compared: list[tuple[bytes, bytes]] = []

    def comparison(supplied: bytes, configured: bytes) -> bool:
        compared.append((supplied, configured))
        return False

    monkeypatch.setattr(security_module.secrets, "compare_digest", comparison)
    configured = settings()
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=None,
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    response = client.post(
        "/retrieve",
        json={"facts": "query"},
        headers={"X-Precedent-API-Key": "attacker-supplied-key"},
    )

    assert response.status_code == 401
    assert compared == [(b"attacker-supplied-key", SERVICE_KEY.encode())]


def test_security_settings_load_from_environment_and_reject_secret_reuse() -> None:
    configured = ApiSettings.from_env(
        {
            "PRECEDENT_API_KEY": SERVICE_KEY,
            "PRECEDENT_METRICS_KEY": METRICS_KEY,
            "MAX_FACTS_CHARS": "123",
            "MAX_PRINCIPLE_CHARS": "45",
            "MAX_INPUT_TOKENS": "2048",
            "MAX_CONCURRENT_GENERATIONS": "1",
            "PROVIDER_TIMEOUT_SECONDS": "12.5",
            "ENABLE_DOCS": "false",
            "ALLOWED_HOSTS": "api.example.test,localhost",
        }
    )

    assert configured.precedent_api_key == SecretStr(SERVICE_KEY)
    assert configured.metrics_api_key == SecretStr(METRICS_KEY)
    assert configured.max_facts_chars == 123
    assert configured.max_principle_chars == 45
    assert configured.max_input_tokens == 2048
    assert configured.max_concurrent_generations == 1
    assert configured.provider_timeout_seconds == 12.5
    assert configured.enable_docs is False
    assert configured.allowed_hosts == ("api.example.test", "localhost")

    with pytest.raises(ValueError, match="must be distinct"):
        ApiSettings(
            precedent_api_key=SecretStr(SERVICE_KEY),
            metrics_api_key=SecretStr(SERVICE_KEY),
        )


def test_protected_routes_fail_closed_when_credentials_are_not_configured() -> None:
    configured = ApiSettings(model_id="test-model", top_k_default=2, max_top_k=3)
    retriever = FakeRetriever()
    provider = FakeProvider()
    service = RAGApplicationService(
        settings=configured,
        retriever=retriever,
        provider=provider,
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    retrieve = client.post("/retrieve", json={"facts": "query"})
    metrics = client.get("/metrics")

    assert retrieve.status_code == 503
    assert retrieve.json()["error"]["code"] == "authentication_not_configured"
    assert metrics.status_code == 503
    assert retriever.calls == []
    assert provider.calls == []


def test_docs_can_be_disabled_without_hiding_public_status_routes() -> None:
    client, _, _ = client_for(settings_overrides={"enable_docs": False})

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_trusted_host_and_safe_response_headers_are_enforced() -> None:
    client, _, _ = client_for()

    rejected = client.get("/health", headers={"Host": "attacker.example"})
    health = client.get("/health")

    assert rejected.status_code == 400
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert health.headers["Cache-Control"] == "no-store"
    assert "strict-transport-security" not in health.headers
    assert "access-control-allow-origin" not in health.headers


def test_absolute_and_configured_input_limits_reject_before_retrieval() -> None:
    absolute_client, absolute_retriever, _ = client_for()
    configured_client, configured_retriever, _ = client_for(
        settings_overrides={"max_facts_chars": 8, "max_principle_chars": 6}
    )

    absolute = absolute_client.post("/retrieve", json={"facts": "x" * 4001})
    facts = configured_client.post("/retrieve", json={"facts": "ninechars"})
    principle = configured_client.post(
        "/retrieve", json={"facts": "valid", "principle": "too-long"}
    )

    for response in (absolute, facts, principle):
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"
    assert absolute_retriever.calls == []
    assert configured_retriever.calls == []


def test_context_budget_rejects_before_provider_execution() -> None:
    passage = "bounded evidence text " * 2000
    retriever = FakeRetriever(
        evidence=(evidence_item("E1", case_id="case:941", passage=passage, rank=1),)
    )
    provider = FakeProvider()
    client, _, _ = client_for(
        retriever=retriever,
        provider=provider,
        settings_overrides={"max_input_tokens": 512},
    )

    response = client.post("/answer", json={"facts": "query", "top_k": 1})
    metrics = client.get("/metrics")

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "context_budget_exceeded"
    assert provider.calls == []
    assert (
        metric_value(
            metrics.text,
            "precedent_failures_total",
            category="context_budget_exceeded",
        )
        == 1
    )


def test_generation_concurrency_limit_is_non_queuing_and_exception_safe() -> None:
    configured = settings(max_concurrent_generations=1)
    provider = FakeProvider(behavior="timeout")
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=provider,
    )
    client = OfflineASGIClient(
        create_app(settings=configured, service=service),
        headers={
            "X-Precedent-API-Key": SERVICE_KEY,
            "X-Precedent-Metrics-Key": METRICS_KEY,
        },
    )

    assert service.try_acquire_generation_slot() is True
    saturated = client.post("/answer", json={"facts": "query"})
    service.release_generation_slot()
    timed_out = client.post("/answer", json={"facts": "query"})
    repeated = client.post("/answer", json={"facts": "query"})
    metrics = client.get("/metrics")

    assert saturated.status_code == 429
    assert saturated.headers["Retry-After"] == "1"
    assert timed_out.status_code == 504
    assert repeated.status_code == 504
    assert len(provider.calls) == 2
    assert metric_value(metrics.text, "precedent_failures_total", category="concurrency_limit") == 1
    assert metric_value(metrics.text, "precedent_failures_total", category="provider_timeout") == 2


def test_secrets_never_enter_responses_logs_metrics_or_openapi(
    caplog: pytest.LogCaptureFixture,
) -> None:
    openai_secret = "sk-security-sentinel-openai"
    service_secret = "security-sentinel-service-key"
    metrics_secret = "security-sentinel-metrics-key"
    configured = ApiSettings(
        model_id="test-model",
        top_k_default=2,
        max_top_k=3,
        openai_api_key=SecretStr(openai_secret),
        precedent_api_key=SecretStr(service_secret),
        metrics_api_key=SecretStr(metrics_secret),
    )
    service = RAGApplicationService(
        settings=configured,
        retriever=FakeRetriever(),
        provider=FakeProvider(behavior="unexpected"),
    )
    client = OfflineASGIClient(create_app(settings=configured, service=service))

    with caplog.at_level("INFO", logger="sg_legal_rag.api"):
        wrong = client.post(
            "/retrieve",
            json={"facts": "query"},
            headers={"X-Precedent-API-Key": service_secret + "-wrong"},
        )
        failed = client.post(
            "/answer",
            json={"facts": "query"},
            headers={"X-Precedent-API-Key": service_secret},
        )
        metrics = client.get("/metrics", headers={"X-Precedent-Metrics-Key": metrics_secret})
        schema = client.get("/openapi.json")

    assert wrong.status_code == 401
    assert failed.status_code == 502
    rendered = "\n".join(
        (wrong.text, failed.text, metrics.text, schema.text)
        + tuple(record.getMessage() for record in caplog.records)
    )
    for secret in (openai_secret, service_secret, metrics_secret):
        assert secret not in rendered
