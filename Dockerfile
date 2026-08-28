# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/precedent
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra api --extra generation

FROM python:3.12-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="Precedent SG RAG" \
      org.opencontainers.image.description="Grounded Singapore precedent retrieval API"
RUN groupadd --gid 10001 precedent \
    && useradd --uid 10001 --gid precedent --no-create-home --shell /usr/sbin/nologin precedent
WORKDIR /opt/precedent
COPY --from=builder --chown=10001:10001 /opt/precedent/.venv ./.venv
ENV PATH="/opt/precedent/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRECEDENT_RETRIEVAL_ARTIFACTS=/opt/precedent/retrieval-artifacts
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]
CMD ["uvicorn", "sg_legal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
