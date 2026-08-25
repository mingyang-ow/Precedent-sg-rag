from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .bm25 import Ranking

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def _deterministic_top_positions(scores: FloatArray, candidate_ids: IntArray, k: int) -> IntArray:
    if k >= len(scores):
        selected = np.arange(len(scores), dtype=np.int64)
    else:
        partition = np.argpartition(-scores, k - 1)[:k]
        threshold = scores[partition].min()
        above = np.flatnonzero(scores > threshold)
        equal = np.flatnonzero(scores == threshold)
        equal = equal[np.argsort(candidate_ids[equal], kind="stable")]
        selected = np.concatenate((above, equal[: k - len(above)]))
    order = np.lexsort((candidate_ids[selected], -scores[selected]))
    return selected[order]


def ranking_from_scores(
    scores: FloatArray,
    candidate_ids: IntArray,
    relevant_ids: set[int],
    *,
    top_k: int,
    candidate_ids_validated: bool = False,
) -> Ranking:
    if scores.ndim != 1 or candidate_ids.ndim != 1 or len(scores) != len(candidate_ids):
        raise ValueError("scores and candidate IDs must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("dense scores must be finite")
    if not relevant_ids:
        raise ValueError("at least one relevant candidate is required")
    if not candidate_ids_validated:
        if len(np.unique(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        candidate_set = {int(identifier) for identifier in candidate_ids}
        if not relevant_ids <= candidate_set:
            raise ValueError("all relevant IDs must occur in the candidate set")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    relevant_mask = np.isin(candidate_ids, np.fromiter(relevant_ids, dtype=np.int64))
    best_relevant_score = scores[relevant_mask].max()
    best_relevant_id = int(candidate_ids[relevant_mask & (scores == best_relevant_score)].min())
    first_relevant_rank = (
        1
        + int(np.count_nonzero(scores > best_relevant_score))
        + int(
            np.count_nonzero((scores == best_relevant_score) & (candidate_ids < best_relevant_id))
        )
    )
    top_positions = _deterministic_top_positions(
        scores, candidate_ids, min(top_k, len(candidate_ids))
    )
    return Ranking(
        top_indices=tuple(int(identifier) for identifier in candidate_ids[top_positions]),
        first_relevant_rank=first_relevant_rank,
        positive_matches=len(scores),
    )


def normalize_rows(values: FloatArray) -> FloatArray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero embedding")
    return np.asarray(values / norms, dtype=np.float32)


def relevant_sets_to_arrays(relevant_sets: Iterable[set[int]]) -> list[IntArray]:
    return [np.fromiter(sorted(values), dtype=np.int64) for values in relevant_sets]
