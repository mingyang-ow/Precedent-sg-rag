from __future__ import annotations

import json
import secrets
from typing import Annotated

import tiktoken
from fastapi import Request, Security
from fastapi.security import APIKeyHeader
from pydantic import SecretStr

from sg_legal_rag.generation.evidence import EvidencePackage
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_SYSTEM_INSTRUCTIONS,
    ProductionAnswer,
    render_production_user_input,
)

TOKEN_OVERHEAD_PER_REQUEST = 32
SERVICE_API_KEY_HEADER = APIKeyHeader(
    name="X-Precedent-API-Key",
    scheme_name="PrecedentServiceKey",
    description="Runtime service credential required by retrieval and answer operations.",
    auto_error=False,
)
METRICS_API_KEY_HEADER = APIKeyHeader(
    name="X-Precedent-Metrics-Key",
    scheme_name="PrecedentMetricsKey",
    description="Separate runtime credential required by the internal metrics endpoint.",
    auto_error=False,
)


class AuthenticationFailed(RuntimeError):
    pass


class AuthenticationNotConfigured(RuntimeError):
    pass


class RequestTooLarge(ValueError):
    pass


class ContextBudgetExceeded(ValueError):
    pass


class GenerationConcurrencyExceeded(RuntimeError):
    pass


def _verify_credential(provided: str | None, expected: SecretStr | None) -> None:
    if expected is None:
        raise AuthenticationNotConfigured("required service credential is not configured")
    supplied = (provided or "").encode("utf-8")
    configured = expected.get_secret_value().encode("utf-8")
    if not secrets.compare_digest(supplied, configured):
        raise AuthenticationFailed("service credential is missing or invalid")


def require_service_auth(
    request: Request,
    credential: Annotated[str | None, Security(SERVICE_API_KEY_HEADER)],
) -> None:
    _verify_credential(credential, request.app.state.settings.precedent_api_key)


def require_metrics_auth(
    request: Request,
    credential: Annotated[str | None, Security(METRICS_API_KEY_HEADER)],
) -> None:
    _verify_credential(credential, request.app.state.settings.metrics_api_key)


def estimate_production_input_tokens(package: EvidencePackage, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    schema = json.dumps(
        ProductionAnswer.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        len(encoding.encode(PRODUCTION_SYSTEM_INSTRUCTIONS))
        + len(encoding.encode(render_production_user_input(package)))
        + len(encoding.encode(schema))
        + TOKEN_OVERHEAD_PER_REQUEST
    )
