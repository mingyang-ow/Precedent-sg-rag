from __future__ import annotations

import logging
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

from .models import ErrorDetail, ErrorResponse
from .observability import current_request_id, install_request_middleware, log_event
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

LOGGER = logging.getLogger("sg_legal_rag.api")


def _error(status_code: int, code: str, message: str, issues: tuple[str, ...] = ()) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=current_request_id(),
            issues=issues,
        )
    )
    log_event("http_error", status=status_code, error_code=code)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def create_app(
    *,
    settings: ApiSettings | None = None,
    service: RAGApplicationService | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_env()
    resolved_service = service or build_default_service(resolved_settings)

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
    install_request_middleware(application)
    application.include_router(router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        issue_types = tuple(str(issue.get("type", "validation_error")) for issue in error.errors())
        return _error(422, "request_validation_failed", "Request validation failed.", issue_types)

    @application.exception_handler(BadRequestError)
    async def bad_request_handler(request: Request, error: BadRequestError) -> JSONResponse:
        return _error(400, "bad_request", str(error))

    @application.exception_handler(RetrievalUnavailable)
    async def retrieval_unavailable_handler(
        request: Request, error: RetrievalUnavailable
    ) -> JSONResponse:
        return _error(503, "retrieval_unavailable", "Retrieval dependencies are unavailable.")

    @application.exception_handler(GenerationUnavailable)
    async def generation_unavailable_handler(
        request: Request, error: GenerationUnavailable
    ) -> JSONResponse:
        return _error(503, "generation_unavailable", "Generation is not configured.")

    @application.exception_handler(ProviderExecutionError)
    async def provider_failure_handler(
        request: Request, error: ProviderExecutionError
    ) -> JSONResponse:
        return _error(502, "provider_failure", "The generation provider failed.")

    @application.exception_handler(MalformedGeneratedOutput)
    async def malformed_output_handler(
        request: Request, error: MalformedGeneratedOutput
    ) -> JSONResponse:
        return _error(502, "malformed_generated_output", "The provider output was invalid.")

    @application.exception_handler(CitationContractViolation)
    async def citation_contract_handler(
        request: Request, error: CitationContractViolation
    ) -> JSONResponse:
        codes = tuple(issue.code.value for issue in error.issues)
        if CitationContractIssueCode.EVIDENCE_DIGEST_MISMATCH in {
            issue.code for issue in error.issues
        }:
            return _error(
                500,
                "evidence_integrity_failure",
                "Stored evidence failed its integrity check.",
                codes,
            )
        return _error(
            502,
            "invalid_generated_citation",
            "The generated answer violated the citation contract.",
            codes,
        )

    @application.exception_handler(EvidenceIntegrityError)
    async def evidence_integrity_handler(
        request: Request, error: EvidenceIntegrityError
    ) -> JSONResponse:
        return _error(500, "evidence_integrity_failure", "Retrieved evidence is inconsistent.")

    @application.exception_handler(Exception)
    async def unexpected_failure_handler(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("unhandled API failure")
        return _error(500, "internal_error", "Internal service failure.")

    return application


app = create_app()
