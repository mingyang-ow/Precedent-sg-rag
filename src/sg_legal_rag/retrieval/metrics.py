from __future__ import annotations

import math
from collections.abc import Iterable

from .bm25 import Ranking


def metrics_for_ranking(
    ranking: Ranking, relevant_indices: set[int], k_values: Iterable[int]
) -> dict[str, float]:
    if not relevant_indices:
        raise ValueError("at least one relevant document is required")
    result = {
        "mrr": 0.0 if ranking.first_relevant_rank is None else 1.0 / ranking.first_relevant_rank
    }
    for k in k_values:
        if k < 1:
            raise ValueError("k values must be positive")
        retrieved = ranking.top_indices[:k]
        hits = sum(index in relevant_indices for index in retrieved)
        result[f"recall_at_{k}"] = hits / len(relevant_indices)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, index in enumerate(retrieved, start=1)
            if index in relevant_indices
        )
        ideal_hits = min(len(relevant_indices), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        result[f"ndcg_at_{k}"] = dcg / ideal_dcg
    return result


def mean_metrics(observations: Iterable[dict[str, float]]) -> dict[str, float]:
    items = list(observations)
    if not items:
        raise ValueError("cannot aggregate an empty observation set")
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}
