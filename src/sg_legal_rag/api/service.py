from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidenceItem,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)
from sg_legal_rag.generation.production_contract import (
    ResolvedProductionAnswer,
    resolve_production_answer,
)

from .provider import (
    OpenAIProductionProvider,
    ProductionGenerationProvider,
)
from .retrieval import EvidenceRetriever, LazyPassageBM25Retriever, RetrievalUnavailable
from .settings import ApiSettings


class BadRequestError(ValueError):
    pass


class GenerationUnavailable(RuntimeError):
    pass


class EvidenceIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceTimings:
    total_ms: float
    retrieval_ms: float
    generation_ms: float = 0.0
    resolution_ms: float = 0.0


@dataclass(frozen=True)
class RetrievalOperation:
    package: EvidencePackage
    timings: ServiceTimings


@dataclass(frozen=True)
class AnswerOperation:
    answer: ResolvedProductionAnswer
    timings: ServiceTimings
    retrieval_count: int
    provider_status: str


class RAGApplicationService:
    def __init__(
        self,
        *,
        settings: ApiSettings,
        retriever: EvidenceRetriever,
        provider: ProductionGenerationProvider | None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.provider = provider

    def readiness(self) -> tuple[bool, bool]:
        return self.retriever.is_ready(), self.provider is not None

    def retrieve(
        self, *, facts: str, principle: str | None, top_k: int | None
    ) -> RetrievalOperation:
        total_started = time.perf_counter()
        effective_top_k = self._effective_top_k(top_k)
        query_mode, query_text = _query_text(facts, principle)
        retrieval_started = time.perf_counter()
        try:
            evidence = self.retriever.retrieve(query_text, top_k=effective_top_k)
        except RetrievalUnavailable:
            raise
        except Exception as error:
            raise RetrievalUnavailable("historical passage retrieval failed") from error
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        package = _production_package(
            query_mode=query_mode,
            query_text=query_text,
            top_k=effective_top_k,
            evidence=evidence,
        )
        return RetrievalOperation(
            package=package,
            timings=ServiceTimings(
                total_ms=(time.perf_counter() - total_started) * 1000,
                retrieval_ms=retrieval_ms,
            ),
        )

    def answer(self, *, facts: str, principle: str | None, top_k: int | None) -> AnswerOperation:
        if self.provider is None:
            raise GenerationUnavailable("generation provider is not configured")
        total_started = time.perf_counter()
        retrieval = self.retrieve(facts=facts, principle=principle, top_k=top_k)
        generation_started = time.perf_counter()
        generation = self.provider.generate(retrieval.package)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        resolution_started = time.perf_counter()
        resolved = resolve_production_answer(retrieval.package, generation.answer)
        resolution_ms = (time.perf_counter() - resolution_started) * 1000
        return AnswerOperation(
            answer=resolved,
            timings=ServiceTimings(
                total_ms=(time.perf_counter() - total_started) * 1000,
                retrieval_ms=retrieval.timings.retrieval_ms,
                generation_ms=generation_ms,
                resolution_ms=resolution_ms,
            ),
            retrieval_count=len(retrieval.package.evidence),
            provider_status=generation.provider_status,
        )

    def _effective_top_k(self, requested: int | None) -> int:
        value = self.settings.top_k_default if requested is None else requested
        if value < 1 or value > self.settings.max_top_k:
            raise BadRequestError(f"top_k must be between 1 and {self.settings.max_top_k}")
        return value


def _query_text(facts: str, principle: str | None) -> tuple[str, str]:
    facts_text = " ".join(facts.split())
    principle_text = " ".join(principle.split()) if principle else None
    return (
        ("facts_principle", f"{facts_text} {principle_text}")
        if principle_text
        else ("facts_only", facts_text)
    )


def _production_package(
    *,
    query_mode: str,
    query_text: str,
    top_k: int,
    evidence: tuple[EvidenceItem, ...],
) -> EvidencePackage:
    identifiers = tuple(item.evidence_id for item in evidence)
    if len(set(identifiers)) != len(identifiers):
        raise EvidenceIntegrityError("retrieval returned duplicate evidence IDs")
    if any(item.origin is not EvidenceOrigin.HISTORICAL_RETRIEVAL for item in evidence):
        raise EvidenceIntegrityError("production retrieval returned non-historical evidence")
    query_payload = json.dumps([query_mode, query_text], ensure_ascii=False, separators=(",", ":"))
    query_id = hashlib.sha256(query_payload.encode("utf-8")).hexdigest()[:20]
    package_payload = json.dumps(
        [
            query_id,
            top_k,
            [[item.evidence_id, item.case_id, item.passage_digest] for item in evidence],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    package_id = hashlib.sha256(package_payload.encode("utf-8")).hexdigest()[:24]
    return EvidencePackage(
        package_id=package_id,
        query_id=query_id,
        query_mode=query_mode,
        query_text=query_text,
        stratum="production_api",
        condition=EvidenceCondition.RETRIEVED,
        top_k=top_k,
        evidence=evidence,
        accepted_case_ids=(),
        warm_start=False,
        target_present=False,
        evidence_sufficient=None,
        expected_action=ExpectedAction.UNKNOWN_NEEDS_REVIEW,
        sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED,
    )


def build_default_service(settings: ApiSettings) -> RAGApplicationService:
    retriever = LazyPassageBM25Retriever(
        data_dir=settings.data_dir,
        splits_path=settings.splits_path,
        config_path=settings.corpus_config_path,
    )
    provider = (
        OpenAIProductionProvider(
            api_key=settings.openai_api_key,
            model_id=settings.model_id,
            max_output_tokens=settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort,
            verbosity=settings.verbosity,
        )
        if settings.openai_api_key is not None
        else None
    )
    return RAGApplicationService(settings=settings, retriever=retriever, provider=provider)
