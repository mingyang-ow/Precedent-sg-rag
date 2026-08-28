from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenUsageLike(Protocol):
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProviderPricing:
    snapshot_date: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


FROZEN_MODEL_PRICING = {
    "gpt-5.6-luna": ProviderPricing(
        snapshot_date="2026-08-26",
        input_usd_per_million=0.20,
        cached_input_usd_per_million=0.02,
        output_usd_per_million=1.20,
    )
}


def pricing_for_model(model: str) -> ProviderPricing | None:
    return FROZEN_MODEL_PRICING.get(model)


def estimated_usage_cost(usage: TokenUsageLike, pricing: ProviderPricing) -> float:
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
    return (
        uncached * pricing.input_usd_per_million
        + usage.cached_input_tokens * pricing.cached_input_usd_per_million
        + usage.output_tokens * pricing.output_usd_per_million
    ) / 1_000_000
