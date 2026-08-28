from __future__ import annotations

import csv
from pathlib import Path

from sg_legal_rag.ingestion.validation import EXPECTED_FIELDS
from sg_legal_rag.retrieval.benchmark import load_full_corpus_and_queries


def test_full_query_loader_groups_multiple_relevant_cases(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    splits = tmp_path / "splits.csv"
    base = {
        "Judgment_URL": "test-url",
        "Judgment_Reference": "[2025] SGCA 1",
        "Year": "2025",
        "Court_Type": "SGCA",
        "Case_Number": "1",
        "Case Name": "A v B [2025] SGCA 1",
        "Current Court Level": "Court of Appeal",
        "Fact_Query": "shared facts",
        "Cited Case": "Case One",
        "Paragraph": "paragraph",
        "Key Principles Illustrated": "shared principle",
        "Issue": "issue",
        "Issue Group": "group",
    }
    with dataset.open("w", encoding="latin-1", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPECTED_FIELDS)
        writer.writeheader()
        writer.writerow(base)
        writer.writerow({**base, "Cited Case": "Case Two"})
    with splits.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Judgment_URL", "Judgment_Reference", "Year", "Split"))
        writer.writerow(("test-url", "[2025] SGCA 1", "2025", "test"))

    corpus, modes = load_full_corpus_and_queries(dataset, splits)

    assert corpus == ["Case One", "Case Two"]
    assert len(modes["facts_only"]) == 1
    assert modes["facts_only"][0].relevant_texts == {"Case One", "Case Two"}
    assert len(modes["facts_principle"]) == 1
