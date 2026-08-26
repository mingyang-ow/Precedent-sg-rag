from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .evidence import EvidencePackage, prompt_evidence
from .schema import GroundedAnswer

SYSTEM_INSTRUCTIONS = """You are a bounded Singapore precedent-selection evaluator.
Use only the supplied evidence. Do not use outside legal knowledge and do not invent a case,
source, proposition, or quotation. Decide whether the supplied passage supports recommending a
precedent as relevant authority for the legal principle, rule, or test raised by the query. Do not
decide whether the client's facts ultimately satisfy that rule or test or predict the outcome.
A passage that identifies a directly applicable precedent or states the relevant legal test is
sufficient. Recommend that precedent even if factual application remains unresolved, and state
that limitation in the explanation. Do not claim the present facts satisfy the test unless the
passage supports that application. Each claim must cite one evidence_id and copy a short verbatim
supporting_quote from that passage. The recommended_case_id must be a case_id in the evidence.
Case identity alone is insufficient. Abstain only when the passage does not support identifying a
relevant precedent or legal proposition because it is absent, unrelated, ambiguous, or too weak.
For abstention, set status to insufficient_evidence, recommended_case_id to null, claims to [], and
explanation to: The supplied evidence does not support a precedent recommendation. This is an
evaluation output, not legal advice."""


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    reasoning_effort: str
    verbosity: str
    max_output_tokens: int = Field(gt=0)
    prompt_version: str
    temperature: float | None = None
    seed: int | None = None
    input_usd_per_million: float = Field(ge=0)
    cached_input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ProviderCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PROVIDER_API_FAILURE = "provider_api_failure"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_model: str
    returned_model: str | None
    response_id: str | None
    generated_at: str
    latency_ms: float = Field(ge=0)
    usage: TokenUsage | None
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    raw_output: str
    answer: GroundedAnswer | None
    error: str | None
    provider_status: ProviderCallStatus | None = None

    @property
    def call_status(self) -> ProviderCallStatus:
        if self.provider_status is not None:
            return self.provider_status
        if self.error is None and self.answer is not None:
            return ProviderCallStatus.SUCCEEDED
        if self.response_id is None:
            return ProviderCallStatus.PROVIDER_API_FAILURE
        return ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE


class GenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_schema: int = 1
    run_signature: str
    package: EvidencePackage
    prompt_version: str
    system_instructions: str
    user_input: str
    settings: GenerationSettings
    result: ProviderResult


class GroundedGenerator(Protocol):
    def generate(
        self, package: EvidencePackage, settings: GenerationSettings
    ) -> ProviderResult: ...


def render_user_input(package: EvidencePackage) -> str:
    visible = {
        "query": {"mode": package.query_mode, "text": package.query_text},
        "evidence": prompt_evidence(package),
    }
    return json.dumps(visible, ensure_ascii=False, indent=2)


def _usage(response: Any) -> TokenUsage | None:
    usage = response.usage
    if usage is None:
        return None
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=int(usage.input_tokens),
        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(usage.output_tokens),
        reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=int(usage.total_tokens),
    )


def estimated_cost(usage: TokenUsage | None, settings: GenerationSettings) -> float | None:
    if usage is None:
        return None
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
    return (
        uncached * settings.input_usd_per_million
        + usage.cached_input_tokens * settings.cached_input_usd_per_million
        + usage.output_tokens * settings.output_usd_per_million
    ) / 1_000_000


class OpenAIResponsesGenerator:
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=0)
        self.client = client

    def generate(self, package: EvidencePackage, settings: GenerationSettings) -> ProviderResult:
        user_input = render_user_input(package)
        started = time.perf_counter()
        response = self.client.responses.parse(
            model=settings.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=user_input,
            text_format=GroundedAnswer,
            text={"verbosity": settings.verbosity},
            max_output_tokens=settings.max_output_tokens,
            reasoning={"effort": settings.reasoning_effort},
            store=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = _usage(response)
        answer = response.output_parsed
        error = None if answer is not None else "response did not contain a parsed answer"
        return ProviderResult(
            requested_model=settings.model,
            returned_model=response.model,
            response_id=response.id,
            generated_at=datetime.now(UTC).isoformat(),
            latency_ms=latency_ms,
            usage=usage,
            estimated_cost_usd=estimated_cost(usage, settings),
            raw_output=response.output_text,
            answer=answer,
            error=error,
            provider_status=(
                ProviderCallStatus.SUCCEEDED
                if answer is not None
                else ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE
            ),
        )


def error_result(
    settings: GenerationSettings,
    started: float,
    error: Exception,
    *,
    status: ProviderCallStatus,
) -> ProviderResult:
    return ProviderResult(
        requested_model=settings.model,
        returned_model=None,
        response_id=None,
        generated_at=datetime.now(UTC).isoformat(),
        latency_ms=(time.perf_counter() - started) * 1000,
        usage=None,
        estimated_cost_usd=None,
        raw_output="",
        answer=None,
        error=f"{type(error).__name__}: {error}",
        provider_status=status,
    )


def exception_call_status(error: Exception) -> ProviderCallStatus:
    if isinstance(error, ValidationError):
        return ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE
    return ProviderCallStatus.PROVIDER_API_FAILURE


def generate_record(
    generator: GroundedGenerator,
    package: EvidencePackage,
    settings: GenerationSettings,
    run_signature: str,
) -> GenerationRecord:
    user_input = render_user_input(package)
    started = time.perf_counter()
    try:
        result = generator.generate(package, settings)
    except Exception as error:  # noqa: BLE001 - provider failures are cached outcomes.
        result = error_result(
            settings,
            started,
            error,
            status=exception_call_status(error),
        )
    return GenerationRecord(
        run_signature=run_signature,
        package=package,
        prompt_version=settings.prompt_version,
        system_instructions=SYSTEM_INSTRUCTIONS,
        user_input=user_input,
        settings=settings,
        result=result,
    )


def cache_path(cache_dir: Path, run_signature: str, package_id: str) -> Path:
    return cache_dir / run_signature / f"{package_id}.json"


def save_record(path: Path, record: GenerationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_record(path: Path, *, run_signature: str, package_id: str) -> GenerationRecord | None:
    if not path.exists():
        return None
    record = GenerationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.run_signature != run_signature:
        raise ValueError(f"cache signature mismatch: {path}")
    if record.package.package_id != package_id:
        raise ValueError(f"cache package mismatch: {path}")
    return record
