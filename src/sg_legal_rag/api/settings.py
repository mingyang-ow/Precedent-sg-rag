from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    data_dir: Path = PROJECT_ROOT / "data/raw"
    splits_path: Path = PROJECT_ROOT / "data/processed/splits_temporal.csv"
    corpus_config_path: Path = PROJECT_ROOT / "configs/corpus_repair.toml"

    @model_validator(mode="after")
    def validate_top_k_policy(self) -> ApiSettings:
        if self.top_k_default > self.max_top_k:
            raise ValueError("TOP_K_DEFAULT cannot exceed MAX_TOP_K")
        return self

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> ApiSettings:
        values = os.environ if environment is None else environment
        api_key = values.get("OPENAI_API_KEY", "").strip()
        payload: dict[str, object] = {
            "model_id": values.get("MODEL_ID", "gpt-5.6-luna"),
            "top_k_default": values.get("TOP_K_DEFAULT", "5"),
            "max_top_k": values.get("MAX_TOP_K", "10"),
            "max_output_tokens": values.get("MAX_OUTPUT_TOKENS", "600"),
            "reasoning_effort": values.get("REASONING_EFFORT", "none"),
            "verbosity": values.get("VERBOSITY", "low"),
            "openai_api_key": SecretStr(api_key) if api_key else None,
            "data_dir": values.get("PRECEDENT_DATA_DIR", str(PROJECT_ROOT / "data/raw")),
            "splits_path": values.get(
                "PRECEDENT_SPLITS_PATH",
                str(PROJECT_ROOT / "data/processed/splits_temporal.csv"),
            ),
            "corpus_config_path": values.get(
                "PRECEDENT_CORPUS_CONFIG",
                str(PROJECT_ROOT / "configs/corpus_repair.toml"),
            ),
        }
        return cls.model_validate(payload)

    @property
    def generation_configured(self) -> bool:
        return self.openai_api_key is not None
