from __future__ import annotations

import math

import pytest

from sg_legal_rag.retrieval.bm25 import BM25Index, Ranking
from sg_legal_rag.retrieval.metrics import metrics_for_ranking
from sg_legal_rag.retrieval.tokenization import tokenize


def test_tokenizer_normalizes_case_and_unicode() -> None:
    assert tokenize("CAFÉ’s Claim [2024] SGCA 1") == ["café’s", "claim", "2024", "sgca", "1"]


def test_bm25_ranks_lexical_match_first() -> None:
    index = BM25Index(["contract damages", "criminal sentence", "equitable damages"])

    ranking = index.rank("contract damages", {0}, top_k=3)

    assert ranking.top_indices[0] == 0
    assert ranking.first_relevant_rank == 1


def test_bm25_uses_deterministic_id_ties() -> None:
    index = BM25Index(["alpha", "beta", "gamma"])

    ranking = index.rank("absent", {2}, top_k=3, candidate_indices=[2, 0, 1])

    assert ranking.top_indices == (0, 1, 2)
    assert ranking.first_relevant_rank == 3


def test_full_corpus_rank_places_zero_score_documents_after_matches() -> None:
    index = BM25Index(["alpha", "beta", "gamma"])

    ranking = index.rank("alpha", {2}, top_k=3)

    assert ranking.top_indices == (0, 1, 2)
    assert ranking.first_relevant_rank == 3


def test_metrics_support_multiple_relevant_documents() -> None:
    ranking = Ranking(top_indices=(2, 0, 1), first_relevant_rank=1, positive_matches=3)

    metrics = metrics_for_ranking(ranking, {1, 2}, (1, 3))

    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["ndcg_at_3"] == pytest.approx(1.5 / (1.0 + 1.0 / math.log2(3)))
