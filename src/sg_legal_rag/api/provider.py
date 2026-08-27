from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr, ValidationError

from sg_legal_rag.generation.evidence import EvidencePackage
from sg_legal_rag.generation.pricing import estimated_usage_cost, pricing_for_model
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_SYSTEM_INSTRUCTIONS,
    ProductionAnswer,
    render_production_user_input,
)
from sg_legal_rag.generation.provider import (
    TokenUsage,
    token_usage_from_response,
)


@dataclass(frozen=True)
class ProductionGenerationResult:
    answer: ProductionAnswer
    provider_status: str
    latency_ms: float
    usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None


class ProductionGenerationProvider(Protocol):
    def generate(self, package: EvidencePackage) -> ProductionGenerationResult: ...


class ProviderExecutionError(RuntimeError):
    pass


class ProviderTimeoutError(ProviderExecutionError):
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
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_family(self) -> str:
        return self._model_id

    def generate(self, package: EvidencePackage) -> ProductionGenerationResult:
        started = time.perf_counter()
        try:
            from openai import APITimeoutError, OpenAI
        except ImportError as error:
            raise ProviderExecutionError("generation provider SDK is unavailable") from error

        try:
            client = OpenAI(
                api_key=self._api_key.get_secret_value(),
                max_retries=0,
                timeout=self._timeout_seconds,
            )
            response = client.responses.parse(
                model=self._model_id,
                instructions=PRODUCTION_SYSTEM_INSTRUCTIONS,
                input=render_production_user_input(package),
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
            usage = token_usage_from_response(response)
            pricing = pricing_for_model(self._model_id)
        except MalformedGeneratedOutput:
            raise
        except APITimeoutError as error:
            raise ProviderTimeoutError("generation provider timed out") from error
        except ValidationError as error:
            raise MalformedGeneratedOutput("provider output violated production schema") from error
        except Exception as error:
            raise ProviderExecutionError("generation provider request failed") from error
        return ProductionGenerationResult(
            answer=answer,
            provider_status="succeeded",
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            estimated_cost_usd=(
                estimated_usage_cost(usage, pricing)
                if usage is not None and pricing is not None
                else None
            ),
        )
