from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from sg_legal_rag.api.app import create_app
from sg_legal_rag.api.provider import (
    MalformedGeneratedOutput,
    ProductionGenerationResult,
    ProviderExecutionError,
)
from sg_legal_rag.api.service import RAGApplicationService
from sg_legal_rag.api.settings import ApiSettings
from sg_legal_rag.generation.evidence import EvidenceItem, EvidenceOrigin, EvidencePackage
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_CITATION_CONTRACT_VERSION,
    ProductionAnswer,
    ProductionClaim,
)
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
        self.calls.append((query_text, top_k))
        return self.evidence[:top_k]


@dataclass
class FakeProvider:
    behavior: str = "answer"
    calls: list[EvidencePackage] = field(default_factory=list)

    def generate(self, package: EvidencePackage) -> ProductionGenerationResult:
        self.calls.append(package)
        if self.behavior == "failure":
            raise ProviderExecutionError("synthetic upstream failure")
        if self.behavior == "malformed":
            raise MalformedGeneratedOutput("synthetic malformed output")
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


def test_health_is_cheap_and_returns_200() -> None:
    client, retriever, _ = client_for()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert retriever.calls == []
    assert response.headers["X-Request-ID"]


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


def test_request_schema_rejects_blank_facts_and_unknown_fields() -> None:
    client, _, _ = client_for()

    blank = client.post("/retrieve", json={"facts": "   "})
    unknown = client.post("/retrieve", json={"facts": "query", "reranker": "hidden-knob"})

    assert blank.status_code == 422
    assert unknown.status_code == 422
    assert blank.json()["error"]["code"] == "request_validation_failed"


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


def test_answer_without_generation_configuration_returns_503_before_retrieval() -> None:
    client, retriever, _ = client_for(provider=None)

    response = client.post("/answer", json={"facts": "query"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "generation_unavailable"
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("behavior", "expected_code"),
    [
        ("invalid_evidence", "invalid_generated_citation"),
        ("case_mismatch", "invalid_generated_citation"),
        ("no_claims", "invalid_generated_citation"),
        ("failure", "provider_failure"),
        ("malformed", "malformed_generated_output"),
    ],
)
def test_answer_maps_generated_and_provider_failures(behavior: str, expected_code: str) -> None:
    client, _, _ = client_for(provider=FakeProvider(behavior=behavior))

    response = client.post("/answer", json={"facts": "query"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


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


def test_openapi_exposes_small_typed_surface_and_contract_metadata() -> None:
    client, _, _ = client_for()

    schema = client.get("/openapi.json").json()
    version = client.get("/version")

    assert {"/health", "/ready", "/version", "/retrieve", "/answer"} <= set(schema["paths"])
    assert "RetrieveRequest" in schema["components"]["schemas"]
    assert "AnswerResponse" in schema["components"]["schemas"]
    assert version.json()["citation_contract"] == "production-citation-v1"
    assert len(version.json()["schema_signature"]) == 64
