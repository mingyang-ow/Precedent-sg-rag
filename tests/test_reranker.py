from __future__ import annotations

import numpy as np

from sg_legal_rag.retrieval.reranker import (
    predict_with_cache,
    ranking_from_candidate_order,
    ranking_from_candidate_scores,
)


def test_reranker_orders_scores_and_breaks_ties_by_candidate_id() -> None:
    ranking = ranking_from_candidate_scores(
        np.asarray([0.5, 0.9, 0.9], dtype=np.float32),
        np.asarray([4, 3, 2], dtype=np.int64),
        {3},
        top_k=3,
    )

    assert ranking.top_indices == (2, 3, 4)
    assert ranking.first_relevant_rank == 2


def test_reranker_returns_no_rank_when_candidate_generation_missed() -> None:
    ranking = ranking_from_candidate_order((1, 2, 3), {9}, top_k=3)

    assert ranking.first_relevant_rank is None


def test_reranker_score_cache_reuses_exact_pair_matrix(tmp_path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, pairs, **_kwargs):
            self.calls += 1
            return np.arange(len(pairs), dtype=np.float32)

    model = FakeModel()
    candidate_ids = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    arguments = {
        "model": model,
        "queries": ["first", "second"],
        "candidate_ids": candidate_ids,
        "corpus": ["alpha", "beta", "gamma"],
        "cache_dir": tmp_path,
        "model_key": "fake",
        "revision": "revision",
        "role": "test",
        "batch_size": 2,
        "max_length": 16,
    }

    created = predict_with_cache(**arguments)
    reused = predict_with_cache(**arguments)

    assert model.calls == 1
    assert created.cache_hit is False
    assert reused.cache_hit is True
    assert reused.values.tolist() == [[0.0, 1.0], [2.0, 3.0]]
