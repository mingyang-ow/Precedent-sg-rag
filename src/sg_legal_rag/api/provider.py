from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr, ValidationError

from sg_legal_rag.generation.evidence import EvidencePackage
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_SYSTEM_INSTRUCTIONS,
    ProductionAnswer,
)
from sg_legal_rag.generation.provider import render_user_input


@dataclass(frozen=True)
class ProductionGenerationResult:
    answer: ProductionAnswer
    provider_status: str
    latency_ms: float


class ProductionGenerationProvider(Protocol):
    def generate(self, package: EvidencePackage) -> ProductionGenerationResult: ...


class ProviderExecutionError(RuntimeError):
    pass


class MalformedGeneratedOutput(RuntimeError):
    pass


class OpenAIProductionProvider:
    """Lazy OpenAI adapter; client construction occurs only inside an answer request."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_id: str,
        max_output_tokens: int,
        reasoning_effort: str,
        verbosity: str,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity

    def generate(self, package: EvidencePackage) -> ProductionGenerationResult:
        started = time.perf_counter()
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key.get_secret_value(), max_retries=0)
            response = client.responses.parse(
                model=self._model_id,
                instructions=PRODUCTION_SYSTEM_INSTRUCTIONS,
                input=render_user_input(package),
                text_format=ProductionAnswer,
                text={"verbosity": self._verbosity},
                max_output_tokens=self._max_output_tokens,
                reasoning={"effort": self._reasoning_effort},
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise MalformedGeneratedOutput("provider returned no parsed production answer")
            answer = (
                parsed
                if isinstance(parsed, ProductionAnswer)
                else ProductionAnswer.model_validate(parsed)
            )
        except MalformedGeneratedOutput:
            raise
        except ValidationError as error:
            raise MalformedGeneratedOutput("provider output violated production schema") from error
        except Exception as error:
            raise ProviderExecutionError("generation provider request failed") from error
        return ProductionGenerationResult(
            answer=answer,
            provider_status="succeeded",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
