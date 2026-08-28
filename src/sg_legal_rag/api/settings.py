from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ABSOLUTE_MAX_FACTS_CHARS = 4000
ABSOLUTE_MAX_PRINCIPLE_CHARS = 2000


class ApiSettings(BaseModel):
    """Typed environment configuration; secrets are never serialized as plaintext."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(default="gpt-5.6-luna", min_length=1)
    top_k_default: int = Field(default=5, ge=1)
    max_top_k: int = Field(default=10, ge=1)
    max_output_tokens: int = Field(default=600, ge=1)
    reasoning_effort: str = Field(default="none", min_length=1)
    verbosity: str = Field(default="low", min_length=1)
    openai_api_key: SecretStr | None = None
    precedent_api_key: SecretStr | None = None
    metrics_api_key: SecretStr | None = None
    max_facts_chars: int = Field(
        default=ABSOLUTE_MAX_FACTS_CHARS, ge=1, le=ABSOLUTE_MAX_FACTS_CHARS
    )
    max_principle_chars: int = Field(
        default=ABSOLUTE_MAX_PRINCIPLE_CHARS,
        ge=1,
        le=ABSOLUTE_MAX_PRINCIPLE_CHARS,
    )
    max_input_tokens: int = Field(default=16_000, ge=512, le=128_000)
    max_concurrent_generations: int = Field(default=2, ge=1, le=32)
    provider_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    enable_docs: bool = True
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    retrieval_artifact_dir: Path = PROJECT_ROOT / "data/processed/retrieval-artifacts"

    @model_validator(mode="after")
    def validate_top_k_policy(self) -> ApiSettings:
        if self.top_k_default > self.max_top_k:
            raise ValueError("TOP_K_DEFAULT cannot exceed MAX_TOP_K")
        secrets = tuple(
            item.get_secret_value()
            for item in (self.openai_api_key, self.precedent_api_key, self.metrics_api_key)
            if item is not None
        )
        if len(secrets) != len(set(secrets)):
            raise ValueError("OpenAI, service, and metrics credentials must be distinct")
        for credential in (self.precedent_api_key, self.metrics_api_key):
            if credential is not None and len(credential.get_secret_value()) < 16:
                raise ValueError("service credentials must contain at least 16 characters")
        if not self.allowed_hosts or any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("ALLOWED_HOSTS must contain explicit non-empty hosts")
        return self

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> ApiSettings:
        values = os.environ if environment is None else environment
        api_key = values.get("OPENAI_API_KEY", "").strip()
        precedent_api_key = values.get("PRECEDENT_API_KEY", "").strip()
        metrics_api_key = values.get("PRECEDENT_METRICS_KEY", "").strip()
        allowed_hosts = tuple(
            host.strip()
            for host in values.get("ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
            if host.strip()
        )
        payload: dict[str, object] = {
            "model_id": values.get("MODEL_ID", "gpt-5.6-luna"),
            "top_k_default": values.get("TOP_K_DEFAULT", "5"),
            "max_top_k": values.get("MAX_TOP_K", "10"),
            "max_output_tokens": values.get("MAX_OUTPUT_TOKENS", "600"),
            "reasoning_effort": values.get("REASONING_EFFORT", "none"),
            "verbosity": values.get("VERBOSITY", "low"),
            "openai_api_key": SecretStr(api_key) if api_key else None,
            "precedent_api_key": SecretStr(precedent_api_key) if precedent_api_key else None,
            "metrics_api_key": SecretStr(metrics_api_key) if metrics_api_key else None,
            "max_facts_chars": values.get("MAX_FACTS_CHARS", str(ABSOLUTE_MAX_FACTS_CHARS)),
            "max_principle_chars": values.get(
                "MAX_PRINCIPLE_CHARS", str(ABSOLUTE_MAX_PRINCIPLE_CHARS)
            ),
            "max_input_tokens": values.get("MAX_INPUT_TOKENS", "16000"),
            "max_concurrent_generations": values.get("MAX_CONCURRENT_GENERATIONS", "2"),
            "provider_timeout_seconds": values.get("PROVIDER_TIMEOUT_SECONDS", "30"),
            "enable_docs": values.get("ENABLE_DOCS", "true"),
            "allowed_hosts": allowed_hosts,
            "retrieval_artifact_dir": values.get(
                "PRECEDENT_RETRIEVAL_ARTIFACTS",
                str(PROJECT_ROOT / "data/processed/retrieval-artifacts"),
            ),
        }
        return cls.model_validate(payload)

    @property
    def generation_configured(self) -> bool:
        return self.openai_api_key is not None
