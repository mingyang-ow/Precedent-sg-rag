from __future__ import annotations

import pytest

from sg_legal_rag.retrieval.hybrid import weighted_reciprocal_rank_fusion


def test_rrf_rewards_candidates_found_by_both_components() -> None:
    ranking = weighted_reciprocal_rank_fusion(
        (((10, 11, 12), 1.0), ((12, 11, 13), 1.0)),
        {12},
        rank_constant=60,
        top_k=4,
    )

    assert ranking.top_indices == (12, 11, 10, 13)
    assert ranking.first_relevant_rank == 1
    assert ranking.positive_matches == 4


def test_rrf_uses_candidate_id_to_break_score_ties() -> None:
    ranking = weighted_reciprocal_rank_fusion(
        (((3,), 1.0), ((2,), 1.0)),
        {3},
        rank_constant=60,
        top_k=2,
    )

    assert ranking.top_indices == (2, 3)
    assert ranking.first_relevant_rank == 2


def test_rrf_returns_no_rank_for_unretrieved_relevant_candidate() -> None:
    ranking = weighted_reciprocal_rank_fusion(
        (((1, 2), 1.0), ((2, 3), 1.0)),
        {9},
        rank_constant=60,
        top_k=3,
    )

    assert ranking.first_relevant_rank is None


def test_rrf_rejects_duplicate_component_candidates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        weighted_reciprocal_rank_fusion(
            (((1, 1), 1.0),),
            {1},
            rank_constant=60,
            top_k=1,
        )
