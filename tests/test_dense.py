from __future__ import annotations

import numpy as np
import pytest

from sg_legal_rag.retrieval.dense import normalize_rows, ranking_from_scores
from sg_legal_rag.retrieval.embedding import encode_with_cache, texts_digest


def test_dense_scores_rank_highest_similarity_first() -> None:
    scores = np.asarray([0.1, 0.9, 0.4], dtype=np.float32)
    candidate_ids = np.asarray([0, 1, 2], dtype=np.int64)

    ranking = ranking_from_scores(scores, candidate_ids, {1}, top_k=3)

    assert ranking.top_indices == (1, 2, 0)
    assert ranking.first_relevant_rank == 1


def test_dense_scores_use_candidate_id_to_break_ties() -> None:
    scores = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    candidate_ids = np.asarray([2, 0, 1], dtype=np.int64)

    ranking = ranking_from_scores(scores, candidate_ids, {2}, top_k=2)

    assert ranking.top_indices == (0, 1)
    assert ranking.first_relevant_rank == 3


def test_dense_scores_use_best_of_multiple_relevant_candidates() -> None:
    scores = np.asarray([0.8, 0.3, 0.6, 0.9], dtype=np.float32)
    candidate_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)

    ranking = ranking_from_scores(scores, candidate_ids, {10, 11}, top_k=4)

    assert ranking.top_indices == (13, 10, 12, 11)
    assert ranking.first_relevant_rank == 2


def test_normalize_rows_returns_unit_vectors() -> None:
    normalized = normalize_rows(np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    assert np.linalg.norm(normalized, axis=1) == pytest.approx([1.0, 1.0])


def test_text_digest_uses_unambiguous_length_prefixes() -> None:
    assert texts_digest(["ab", "c"]) != texts_digest(["a", "bc"])


def test_embedding_cache_reuses_exact_text_rows(tmp_path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **_kwargs):
            self.calls.append(texts)
            return np.asarray([[len(text), len(text) + 1] for text in texts], dtype=np.float32)

    model = FakeModel()
    source_texts = ["alpha", "beta"]
    source = encode_with_cache(
        model,
        source_texts,
        cache_dir=tmp_path,
        model_key="fake",
        revision="revision",
        role="source",
        batch_size=2,
        dimensions=2,
    )

    target = encode_with_cache(
        model,
        ["beta", "gamma"],
        cache_dir=tmp_path,
        model_key="fake",
        revision="revision",
        role="target",
        batch_size=2,
        dimensions=2,
        reuse_texts=source_texts,
        reuse_values=source.values,
    )

    assert model.calls == [["alpha", "beta"], ["gamma"]]
    assert target.reused_rows == 1
    assert target.encoded_rows == 1
    assert target.values.tolist() == [[4.0, 5.0], [5.0, 6.0]]
