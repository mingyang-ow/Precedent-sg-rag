from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sg_legal_rag.generation.production_contract import (
    CitationContractIssueCode,
    CitationContractViolation,
)

from .metrics import ApiMetrics
from .models import ErrorDetail, ErrorResponse
from .observability import (
    current_request_id,
    install_request_middleware,
    log_event,
    route_template,
)
from .provider import MalformedGeneratedOutput, ProviderExecutionError
from .retrieval import RetrievalUnavailable
from .routes import router
from .service import (
    BadRequestError,
    EvidenceIntegrityError,
    GenerationUnavailable,
    RAGApplicationService,
    build_default_service,
)
from .settings import ApiSettings


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    failure_category: str,
    issues: tuple[str, ...] = (),
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=current_request_id(),
            issues=issues,
        )
    )
    metrics: ApiMetrics = request.app.state.metrics
    metrics.record_failure(failure_category)
    endpoint = route_template(request)
    if endpoint in {"/retrieve", "/answer"}:
        metrics.record_rag_operation(operation=endpoint.removeprefix("/"), outcome="failure")
    log_event(
        "http_error",
        endpoint=endpoint,
        status=status_code,
        error_code=code,
        issue_codes=issues,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def create_app(
    *,
    settings: ApiSettings | None = None,
    service: RAGApplicationService | None = None,
    metrics: ApiMetrics | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_env()
    resolved_metrics = metrics or ApiMetrics()
    resolved_service = service or build_default_service(resolved_settings)
    resolved_service.install_metrics(resolved_metrics)

    @asynccontextmanager
    async def lifespan(current_app: FastAPI) -> AsyncIterator[None]:
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="precedent-api")
        current_app.state.executor = executor
        try:
            yield
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(
        title="Precedent SG RAG API",
        summary="Evidence-traceable Singapore precedent retrieval and grounded answers",
        description=(
            "A production-style API where models reference bounded evidence and the application "
            "owns authoritative source text. This service is not legal advice."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.rag_service = resolved_service
    application.state.metrics = resolved_metrics
    install_request_middleware(application, resolved_metrics)
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        issue_types = tuple(str(issue.get("type", "validation_error")) for issue in error.errors())
        return _error(
            request,
            422,
            "request_validation_failed",
            "Request validation failed.",
            "request_validation_failure",
            issue_types,
        )

    @application.exception_handler(BadRequestError)
    async def bad_request_handler(request: Request, error: BadRequestError) -> JSONResponse:
        return _error(
            request,
            400,
            "bad_request",
            str(error),
            "request_validation_failure",
        )

    @application.exception_handler(RetrievalUnavailable)
    async def retrieval_unavailable_handler(
        request: Request, error: RetrievalUnavailable
    ) -> JSONResponse:
        return _error(
            request,
            503,
            "retrieval_unavailable",
            "Retrieval dependencies are unavailable.",
            "retrieval_unavailable",
        )

    @application.exception_handler(GenerationUnavailable)
    async def generation_unavailable_handler(
        request: Request, error: GenerationUnavailable
    ) -> JSONResponse:
        return _error(
            request,
            503,
            "generation_unavailable",
            "Generation is not configured.",
            "generation_unavailable",
        )

    @application.exception_handler(ProviderExecutionError)
    async def provider_failure_handler(
        request: Request, error: ProviderExecutionError
    ) -> JSONResponse:
        return _error(
            request,
            502,
            "provider_failure",
            "The generation provider failed.",
            "provider_failure",
        )

    @application.exception_handler(MalformedGeneratedOutput)
    async def malformed_output_handler(
        request: Request, error: MalformedGeneratedOutput
    ) -> JSONResponse:
        return _error(
            request,
            502,
            "malformed_generated_output",
            "The provider output was invalid.",
            "malformed_generated_output",
        )

    @application.exception_handler(CitationContractViolation)
    async def citation_contract_handler(
        request: Request, error: CitationContractViolation
    ) -> JSONResponse:
        codes = tuple(issue.code.value for issue in error.issues)
        request.app.state.metrics.record_citation_violations(codes)
        if CitationContractIssueCode.EVIDENCE_DIGEST_MISMATCH in {
            issue.code for issue in error.issues
        }:
            return _error(
                request,
                500,
                "evidence_integrity_failure",
                "Stored evidence failed its integrity check.",
                "evidence_integrity_failure",
                codes,
            )
        return _error(
            request,
            502,
            "invalid_generated_citation",
            "The generated answer violated the citation contract.",
            "citation_contract_failure",
            codes,
        )

    @application.exception_handler(EvidenceIntegrityError)
    async def evidence_integrity_handler(
        request: Request, error: EvidenceIntegrityError
    ) -> JSONResponse:
        return _error(
            request,
            500,
            "evidence_integrity_failure",
            "Retrieved evidence is inconsistent.",
            "evidence_integrity_failure",
        )

    @application.exception_handler(Exception)
    async def unexpected_failure_handler(request: Request, error: Exception) -> JSONResponse:
        log_event("unhandled_api_failure")
        return _error(
            request,
            500,
            "internal_error",
            "Internal service failure.",
            "internal_error",
        )

    return application


app = create_app()
