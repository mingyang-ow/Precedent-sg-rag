from __future__ import annotations

import math
from collections.abc import Sequence

from .bm25 import Ranking


def weighted_reciprocal_rank_fusion(
    components: Sequence[tuple[Sequence[int], float]],
    relevant_indices: set[int],
    *,
    rank_constant: int,
    top_k: int,
) -> Ranking:
    """Fuse truncated component rankings with deterministic weighted RRF."""
    if not components:
        raise ValueError("at least one component ranking is required")
    if not relevant_indices:
        raise ValueError("at least one relevant document is required")
    if rank_constant < 1:
        raise ValueError("rank constant must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    fused_scores: dict[int, float] = {}
    for candidates, weight in components:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("component weights must be finite and positive")
        candidate_ids = tuple(int(candidate) for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("component rankings must not contain duplicate candidates")
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            fused_scores[candidate_id] = fused_scores.get(candidate_id, 0.0) + weight / (
                rank_constant + rank
            )

    if not fused_scores:
        raise ValueError("component rankings must retrieve at least one candidate")
    ranked = sorted(fused_scores, key=lambda index: (-fused_scores[index], index))
    first_relevant_rank = next(
        (rank for rank, index in enumerate(ranked, start=1) if index in relevant_indices),
        None,
    )
    return Ranking(
        top_indices=tuple(ranked[:top_k]),
        first_relevant_rank=first_relevant_rank,
        positive_matches=len(ranked),
    )
