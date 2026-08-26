from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sg_legal_rag.ingestion.validation import EXPECTED_FIELDS
from sg_legal_rag.retrieval.benchmark import QueryRecord
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.checkpoint import BenchmarkCheckpoint, empty_mode_state
from sg_legal_rag.retrieval.corpus_repair import (
    HistoricalContext,
    build_case_profile,
    canonical_case_key,
    load_corpus_repair_dataset,
    max_aggregate_passage_scores,
    max_aggregate_sparse_passage_scores,
)
from sg_legal_rag.retrieval.corpus_repair_benchmark import evaluate_bm25_mode
from sg_legal_rag.retrieval.dense import ranking_from_scores
from sg_legal_rag.retrieval.metrics import metrics_for_ranking


def row(**overrides: str) -> dict[str, str]:
    values = {
        "Judgment_URL": "historical-url",
        "Judgment_Reference": "[2023] SGCA 1",
        "Year": "2023",
        "Court_Type": "SGCA",
        "Case_Number": "1",
        "Case Name": "A v B [2023] SGCA 1",
        "Current Court Level": "Court of Appeal",
        "Fact_Query": "contract interpretation facts",
        "Cited Case": "Alpha v Beta [2020] SGCA 2",
        "Paragraph": "The court applied Alpha v Beta [2020] SGCA 2 to interpret the contract.",
        "Key Principles Illustrated": "contracts are interpreted objectively",
        "Issue": "contract interpretation",
        "Issue Group": "contract",
    }
    values.update(overrides)
    return values


def write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="latin-1", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPECTED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_splits(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Judgment_URL", "Judgment_Reference", "Year", "Split"))
        writer.writerow(("historical-url", "[2023] SGCA 1", "2023", "validation"))
        writer.writerow(("test-url", "[2024] SGCA 3", "2024", "test"))


def load_fixture(tmp_path: Path, rows: list[dict[str, str]]):
    dataset = tmp_path / "dataset.csv"
    splits = tmp_path / "splits.csv"
    write_dataset(dataset, rows)
    write_splits(splits)
    return load_corpus_repair_dataset(
        dataset,
        splits,
        evidence_cutoff_year=2023,
        max_passage_chars=200,
        max_profile_passages=3,
        max_profile_identifier_chars=100,
        max_profile_context_chars=80,
        max_profile_chars=300,
    )


def context(year: int, url: str, text: str) -> HistoricalContext:
    return HistoricalContext(
        case_key="alpha",
        raw_case="Alpha",
        source_url=url,
        source_reference=f"[{year}] SGCA 1",
        source_year=year,
        text=text,
        original_chars=len(text),
        identifier_matched="alpha" in text.casefold(),
        digest=url,
    )


def test_historical_context_excludes_test_judgment_and_future_year(tmp_path: Path) -> None:
    dataset = load_fixture(
        tmp_path,
        [
            row(),
            row(
                Judgment_URL="test-url",
                Judgment_Reference="[2024] SGCA 3",
                Year="2024",
                Paragraph="Leaked 2024 discussion of Alpha v Beta [2020] SGCA 2.",
            ),
        ],
    )

    assert len(dataset.contexts) == 1
    assert dataset.contexts[0].source_year == 2023
    assert dataset.contexts[0].source_url == "historical-url"
    assert "Leaked" not in dataset.profiles[0]
    gold_rows = dataset.queries_by_mode["facts_only"][0].gold_contexts.values()
    gold = next(iter(gold_rows))
    assert gold.source_url == "test-url"
    assert gold.source_year == 2024
    assert "Leaked 2024 discussion" in gold.paragraph
    assert gold.case_key in dataset.queries_by_mode["facts_only"][0].relevant_texts


def test_gold_context_preserves_exact_test_row_provenance(tmp_path: Path) -> None:
    test_paragraph = (
        "The test court applied Alpha v Beta [2020] SGCA 2 to the exact contract dispute."
    )
    dataset = load_fixture(
        tmp_path,
        [
            row(),
            row(
                Judgment_URL="test-url",
                Judgment_Reference="[2024] SGCA 3",
                Year="2024",
                Paragraph=test_paragraph,
            ),
        ],
    )
    query = dataset.queries_by_mode["facts_only"][0]
    gold = next(iter(query.gold_contexts.values()))

    assert gold.fact_query == query.text
    assert gold.paragraph == test_paragraph
    assert gold.raw_case == "Alpha v Beta [2020] SGCA 2"
    assert gold.identifier_matched
    assert gold.source_reference == "[2024] SGCA 3"
    assert all(context.source_url != gold.source_url for context in dataset.contexts)


def test_duplicate_historical_context_is_removed(tmp_path: Path) -> None:
    dataset = load_fixture(tmp_path, [row(), row()])

    assert len(dataset.contexts) == 1
    assert dataset.audit["duplicate_historical_contexts_removed"] == 1


def test_canonical_case_key_merges_only_orthographic_variants() -> None:
    assert canonical_case_key("  Alpha   v Beta [2020] SGCA 2 ") == canonical_case_key(
        "alpha v beta [2020] sgca 2"
    )
    assert canonical_case_key("Alpha v Beta") != canonical_case_key("[2020] SGCA 2")


def test_profile_is_deterministic_and_respects_all_limits() -> None:
    contexts = [
        context(2021, "b", "Earlier Alpha context " + "x" * 80),
        context(2023, "a", "Newest Alpha context " + "y" * 80),
        context(2022, "c", "Middle Alpha context " + "z" * 80),
    ]
    ordered = sorted(contexts, key=lambda item: (-item.source_year, item.source_url))

    profile, included = build_case_profile(
        "Alpha",
        ordered,
        max_passages=2,
        max_identifier_chars=40,
        max_context_chars=45,
        max_total_chars=150,
    )

    assert included == 2
    assert "Newest Alpha context" in profile
    assert "Middle Alpha context" in profile
    assert "Earlier Alpha context" not in profile
    assert len(profile) <= 150


def test_passage_scores_aggregate_to_case_by_maximum() -> None:
    case_ids, scores = max_aggregate_passage_scores(
        np.asarray([0.2, 0.9, 0.6], dtype=np.float32),
        np.asarray([4, 4, 8], dtype=np.int64),
    )

    assert case_ids.tolist() == [4, 8]
    assert scores.tolist() == pytest.approx([0.9, 0.6])


def test_sparse_passage_aggregation_keeps_deterministic_case_order() -> None:
    scores = max_aggregate_sparse_passage_scores(
        {0: 0.2, 1: 0.9},
        np.asarray([1, 0, 1], dtype=np.int64),
        2,
    )

    assert scores.tolist() == pytest.approx([0.9, 0.2])


def test_warm_and_cold_targets_are_reported_separately(tmp_path: Path) -> None:
    dataset = load_fixture(
        tmp_path,
        [
            row(),
            row(
                Judgment_URL="test-url",
                Judgment_Reference="[2024] SGCA 3",
                Year="2024",
            ),
            row(
                **{
                    "Judgment_URL": "test-url",
                    "Judgment_Reference": "[2024] SGCA 3",
                    "Year": "2024",
                    "Cited Case": "Cold v Start [2023] SGHC 9",
                }
            ),
        ],
    )
    coverage = dataset.audit["coverage"]

    assert coverage["unique_test_targets"] == 2
    assert coverage["warm_start_unique_targets"] == 1
    assert coverage["cold_start_unique_targets"] == 1
    assert coverage["modes"]["facts_only"]["warm_start_queries"] == 1


def test_missing_cold_case_receives_zero_metrics_without_metric_corruption() -> None:
    ranking = ranking_from_scores(
        np.asarray([0.8, 0.4], dtype=np.float32),
        np.asarray([1, 2], dtype=np.int64),
        {9},
        top_k=2,
        allow_missing_relevant=True,
    )

    assert metrics_for_ranking(ranking, {9}, (1, 2)) == {
        "mrr": 0.0,
        "recall_at_1": 0.0,
        "ndcg_at_1": 0.0,
        "recall_at_2": 0.0,
        "ndcg_at_2": 0.0,
    }


def test_benchmark_checkpoint_resumes_without_changing_metrics(tmp_path: Path) -> None:
    documents = ["alpha contract", "beta crime"]
    document_case_ids = np.asarray([0, 1], dtype=np.int64)
    queries = [
        QueryRecord("alpha", {"alpha"}, {"SGCA"}, {"2024"}),
        QueryRecord("beta", {"beta"}, {"SGHC"}, {"2025"}),
    ]
    index = BM25Index(documents)
    case_to_id = {"alpha": 0, "beta": 1}

    full_checkpoint = BenchmarkCheckpoint.load(tmp_path / "full.json", "same", 1)
    full = evaluate_bm25_mode(
        "facts_only",
        index,
        document_case_ids,
        queries,
        case_to_id,
        frozenset({0, 1}),
        (1, 2),
        full_checkpoint,
    )

    partial_state = empty_mode_state()
    full_state = full_checkpoint.state_for("facts_only")
    partial_state["queries_completed"] = 1
    for key in (
        "observations",
        "warm_observations",
        "latencies_ms",
        "warm_latencies_ms",
        "relevant_counts",
        "warm_relevant_counts",
    ):
        partial_state[key] = full_state[key][:1]
    partial_checkpoint = BenchmarkCheckpoint.load(tmp_path / "partial.json", "same", 1)
    partial_checkpoint.modes["facts_only"] = partial_state
    partial_checkpoint.save()

    resumed_checkpoint = BenchmarkCheckpoint.load(tmp_path / "partial.json", "same", 1)
    resumed = evaluate_bm25_mode(
        "facts_only",
        index,
        document_case_ids,
        queries,
        case_to_id,
        frozenset({0, 1}),
        (1, 2),
        resumed_checkpoint,
    )

    assert resumed["all_queries"]["metrics"] == full["all_queries"]["metrics"]
    assert resumed["warm_start_queries"]["metrics"] == full["warm_start_queries"]["metrics"]
    assert resumed_checkpoint.state_for("facts_only")["queries_completed"] == 2


def test_benchmark_checkpoint_rejects_incompatible_signature(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    BenchmarkCheckpoint.load(path, "first", 10).save()

    with pytest.raises(ValueError, match="signature mismatch"):
        BenchmarkCheckpoint.load(path, "second", 10)
