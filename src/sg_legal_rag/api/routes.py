from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from fastapi import APIRouter, Request, Response, Security
from prometheus_client import CONTENT_TYPE_LATEST

from sg_legal_rag.generation.production_contract import (
    PRODUCTION_CITATION_CONTRACT_VERSION,
    PRODUCTION_PROMPT_VERSION,
    production_prompt_signature,
    production_schema_signature,
)

from .metrics import ApiMetrics
from .models import (
    AnswerRequest,
    AnswerResponse,
    ErrorResponse,
    EvidenceResponse,
    HealthResponse,
    LatencyBreakdown,
    ReadinessResponse,
    RetrieveRequest,
    RetrieveResponse,
    VersionResponse,
)
from .observability import current_request_id, log_event
from .security import (
    GenerationConcurrencyExceeded,
    require_metrics_auth,
    require_service_auth,
)
from .service import RAGApplicationService, ServiceTimings

router = APIRouter()
ResultT = TypeVar("ResultT")
ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing or invalid service credential"},
    400: {"model": ErrorResponse, "description": "Invalid request policy"},
    413: {"model": ErrorResponse, "description": "Request or context budget exceeded"},
    429: {"model": ErrorResponse, "description": "Generation capacity saturated"},
    422: {"model": ErrorResponse, "description": "Request schema validation failed"},
    500: {"model": ErrorResponse, "description": "Evidence integrity or internal failure"},
    502: {"model": ErrorResponse, "description": "Provider or generated-output failure"},
    503: {"model": ErrorResponse, "description": "Required dependency unavailable"},
    504: {"model": ErrorResponse, "description": "Generation provider timeout"},
}


def _service(request: Request) -> RAGApplicationService:
    return request.app.state.rag_service


def _metrics(request: Request) -> ApiMetrics:
    return request.app.state.metrics


def _timings(value: ServiceTimings) -> LatencyBreakdown:
    return LatencyBreakdown(
        total_ms=value.total_ms,
        retrieval_ms=value.retrieval_ms,
        generation_ms=value.generation_ms,
        resolution_ms=value.resolution_ms,
    )


async def _run_blocking(
    request: Request,
    operation: Callable[..., ResultT],
    **kwargs: Any,
) -> ResultT:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(request.app.state.executor, partial(operation, **kwargs))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process liveness",
    description="Returns process liveness without checking retrieval or model dependencies.",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/metrics",
    response_class=Response,
    dependencies=[Security(require_metrics_auth)],
    summary="Prometheus metrics",
    description=(
        "Exposes privacy-safe operational metrics. This endpoint should be network-restricted "
        "to internal monitoring systems in production."
    ),
)
async def metrics(request: Request) -> Response:
    return Response(
        content=_metrics(request).render(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    summary="Dependency readiness",
    description=(
        "Reports partial readiness when retrieval is available without generation credentials. "
        "Never calls the model provider."
    ),
)
async def ready(request: Request, response: Response) -> ReadinessResponse:
    retrieval, generation = _service(request).readiness()
    _metrics(request).set_readiness(retrieval=retrieval, generation=generation)
    if not retrieval:
        response.status_code = 503
        status = "not_ready"
    elif not generation:
        status = "partial"
    else:
        status = "ready"
    return ReadinessResponse(
        status=status,
        retrieval=retrieval,
        generation_configured=generation,
        answer_ready=retrieval and generation,
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Service and contract versions",
)
async def version(request: Request) -> VersionResponse:
    identity = _service(request).artifact_identity()
    return VersionResponse(
        citation_contract=PRODUCTION_CITATION_CONTRACT_VERSION,
        prompt_version=PRODUCTION_PROMPT_VERSION,
        prompt_signature=production_prompt_signature(),
        schema_signature=production_schema_signature(),
        retrieval_artifact_version=(identity.artifact_version if identity is not None else None),
        retrieval_artifact_digest=(identity.manifest_digest if identity is not None else None),
        retrieval_document_count=(identity.document_count if identity is not None else None),
        retrieval_load_ms=(identity.load_ms if identity is not None else None),
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Security(require_service_auth)],
    responses=ERROR_RESPONSES,
    summary="Retrieve historical precedent evidence",
    description="Runs passage BM25 and returns application-controlled source passages.",
)
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    operation = await _run_blocking(
        request,
        _service(request).retrieve,
        facts=payload.facts,
        principle=payload.principle,
        top_k=payload.top_k,
    )
    package = operation.package
    results = tuple(
        EvidenceResponse(
            package_id=package.package_id,
            query_id=package.query_id,
            evidence_id=item.evidence_id,
            case_id=item.case_id,
            case_name=item.case_name,
            source_judgment=item.source_judgment,
            source_url=item.source_url,
            source_year=item.source_year,
            passage=item.passage,
            passage_digest=item.passage_digest,
            retrieval_rank=item.retrieval_rank,
            retrieval_score=item.retrieval_score,
            origin=item.origin,
        )
        for item in package.evidence
    )
    log_event(
        "rag_operation",
        endpoint="/retrieve",
        status=200,
        retrieval_count=len(results),
        answer_status=None,
        provider_status="not_requested",
        retrieval_ms=round(operation.timings.retrieval_ms, 3),
        generation_ms=0.0,
        resolution_ms=0.0,
    )
    _metrics(request).record_rag_operation(operation="retrieve", outcome="success")
    return RetrieveResponse(
        request_id=current_request_id(),
        package_id=package.package_id,
        query_id=package.query_id,
        results=results,
        timings=_timings(operation.timings),
    )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    dependencies=[Security(require_service_auth)],
    responses=ERROR_RESPONSES,
    summary="Generate an evidence-resolved precedent answer",
    description=(
        "Retrieves bounded evidence, invokes the configured production provider, validates all "
        "model references, and resolves exact source text from application-owned evidence."
    ),
)
async def answer(payload: AnswerRequest, request: Request) -> AnswerResponse:
    service = _service(request)
    if not service.try_acquire_generation_slot():
        raise GenerationConcurrencyExceeded("generation concurrency limit reached")
    try:
        operation = await _run_blocking(
            request,
            service.answer,
            facts=payload.facts,
            principle=payload.principle,
            top_k=payload.top_k,
        )
    finally:
        service.release_generation_slot()
    resolved = operation.answer
    log_event(
        "rag_operation",
        endpoint="/answer",
        status=200,
        retrieval_count=operation.retrieval_count,
        answer_status=resolved.status.value,
        provider_status=operation.provider_status,
        retrieval_ms=round(operation.timings.retrieval_ms, 3),
        generation_ms=round(operation.timings.generation_ms, 3),
        resolution_ms=round(operation.timings.resolution_ms, 3),
    )
    _metrics(request).record_rag_operation(operation="answer", outcome="success")
    _metrics(request).record_answer_outcome(resolved.status.value)
    return AnswerResponse(
        request_id=current_request_id(),
        contract_version=resolved.contract_version,
        package_id=resolved.package_id,
        query_id=resolved.query_id,
        status=resolved.status,
        recommended_case_id=resolved.recommended_case_id,
        explanation=resolved.explanation,
        claims=resolved.claims,
        timings=_timings(operation.timings),
    )
