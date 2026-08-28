from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, Request

LOGGER = logging.getLogger("sg_legal_rag.api")
LOGGER.setLevel(logging.INFO)
REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_request_id: ContextVar[str] = ContextVar("precedent_request_id", default="unknown")


def current_request_id() -> str:
    return _request_id.get()


def request_id_for_header(value: str | None) -> str:
    return value if value and REQUEST_ID_PATTERN.fullmatch(value) else str(uuid.uuid4())


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "request_id": current_request_id(), **fields}
    LOGGER.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def install_request_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next: Any):
        request_id = request_id_for_header(request.headers.get(REQUEST_ID_HEADER))
        token = _request_id.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            log_event(
                "http_request",
                endpoint=request.url.path,
                method=request.method,
                status=status_code,
                latency_ms=round(latency_ms, 3),
            )
            _request_id.reset(token)
