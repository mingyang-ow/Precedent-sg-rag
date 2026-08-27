from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sg_legal_rag.generation.evidence import EvidenceOrigin
from sg_legal_rag.generation.production_contract import ResolvedClaim
from sg_legal_rag.generation.schema import AnswerStatus


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    facts: str = Field(
        min_length=1,
        max_length=4000,
        description="Facts or legal problem used as the primary retrieval query.",
        examples=["The accused relies on diminished responsibility."],
    )
    principle: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional legal principle appended to the facts for assisted retrieval.",
        examples=["Three cumulative requirements for diminished responsibility."],
    )
    top_k: int | None = Field(
        default=None,
        strict=True,
        description="Maximum number of case-level evidence passages to return.",
        examples=[5],
    )

    @field_validator("facts")
    @classmethod
    def reject_blank_facts(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("facts must contain non-whitespace text")
        return value

    @field_validator("principle")
    @classmethod
    def normalize_blank_principle(cls, value: str | None) -> str | None:
        return value if value and value.strip() else None


class RetrieveRequest(QueryRequest):
    """Retrieve bounded, application-controlled historical evidence."""


class AnswerRequest(QueryRequest):
    """Retrieve evidence and request a production-citation-v1 answer."""


class LatencyBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_ms: float = Field(ge=0)
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    resolution_ms: float = Field(ge=0)


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    query_id: str
    evidence_id: str
    case_id: str
    case_name: str
    source_judgment: str
    source_url: str
    source_year: int
    passage: str
    passage_digest: str
    retrieval_rank: int | None
    retrieval_score: float | None
    origin: EvidenceOrigin


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    package_id: str
    query_id: str
    results: tuple[EvidenceResponse, ...]
    timings: LatencyBreakdown


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    contract_version: Literal["production-citation-v1"]
    package_id: str
    query_id: str
    status: AnswerStatus
    recommended_case_id: str | None
    explanation: str
    claims: tuple[ResolvedClaim, ...]
    timings: LatencyBreakdown


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "partial", "not_ready"]
    retrieval: bool
    generation_configured: bool
    answer_ready: bool


class VersionResponse(BaseModel):
    service_version: Literal["api-v1"] = "api-v1"
    citation_contract: Literal["production-citation-v1"] = "production-citation-v1"
    prompt_version: Literal["rag-production-v1"] = "rag-production-v1"
    prompt_signature: str
    schema_signature: str
    retrieval_artifact_version: str | None
    retrieval_artifact_digest: str | None
    retrieval_document_count: int | None
    retrieval_load_ms: float | None
    historical_contract_preserved: bool = True


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    request_id: str
    issues: tuple[str, ...] = ()


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorDetail
