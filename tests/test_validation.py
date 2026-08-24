from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sg_legal_rag.ingestion.validation import (
    EXPECTED_FIELDS,
    DatasetValidationError,
    inspect_csv,
)


def write_dataset(path: Path, rows: list[dict[str, str]], fields=EXPECTED_FIELDS) -> None:
    with path.open("w", encoding="latin-1", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides: str) -> dict[str, str]:
    result = {
        "Judgment_URL": "https://example.test/2024_SGCA_1",
        "Judgment_Reference": "[2024] SGCA 1",
        "Year": "2024",
        "Court_Type": "SGCA",
        "Case_Number": "1",
        "Case Name": "Alpha v Beta [2024] SGCA 1",
        "Current Court Level": "Singapore Court of Appeal",
        "Fact_Query": "A payment was disputed.",
        "Cited Case": "Gamma v Delta [2020] SGCA 2",
        "Paragraph": "The court considered Gamma.",
        "Key Principles Illustrated": "A payment made by mistake may be recoverable.",
        "Issue": "Whether the payment is recoverable",
        "Issue Group": "Restitution",
    }
    result.update(overrides)
    return result


def test_inspect_csv_reports_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    write_dataset(dataset, [row(), row(**{"Cited Case": "Other v Case [2019] SGHC 4"})])

    report = inspect_csv(dataset)

    assert report.rows == 2
    assert report.unique_judgments == 1
    assert report.unique_cited_cases == 2
    assert report.year_min == report.year_max == 2024
    assert report.inconsistent_judgment_metadata == 0


def test_inspect_csv_rejects_schema_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    write_dataset(dataset, [row()], fields=EXPECTED_FIELDS[:-1])

    with pytest.raises(DatasetValidationError, match="schema mismatch"):
        inspect_csv(dataset)


def test_inspect_csv_quarantines_missing_required_value(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    write_dataset(dataset, [row(Fact_Query="")])

    report = inspect_csv(dataset)

    assert report.rows_eligible_for_all_query_modes == 0
    assert report.missing_by_field["Fact_Query"] == 1
    assert report.quality_issue_samples[0]["missing_required_fields"] == ["Fact_Query"]
