from __future__ import annotations

from sg_legal_rag.ingestion.splits import (
    Judgment,
    audit_assignments,
    grouped_random_assignments,
    normalize_case_family,
    temporal_assignments,
)


def judgments() -> dict[str, Judgment]:
    return {
        "train": Judgment("train", "[2021] SGHC 1", 2021, "Alpha v Beta [2021] SGHC 1"),
        "validation": Judgment("validation", "[2023] SGCA 2", 2023, "Gamma v Delta [2023] SGCA 2"),
        "test": Judgment("test", "[2025] SGHC 3", 2025, "Alpha v Beta [2025] SGHC 3"),
    }


def test_temporal_boundaries() -> None:
    assert temporal_assignments(judgments()) == {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }


def test_grouped_random_assignment_keeps_each_judgment_whole() -> None:
    assignments = grouped_random_assignments(judgments(), seed=42)

    assert set(assignments) == set(judgments())
    assert set(assignments.values()) <= {"train", "validation", "test"}


def test_audit_detects_family_and_target_overlap_without_url_overlap() -> None:
    items = judgments()
    assignments = temporal_assignments(items)
    targets = {"train": ["Shared"], "validation": ["Other"], "test": ["Shared"]}

    audit = audit_assignments("temporal", items, targets, assignments)

    assert audit.judgment_url_overlap == 0
    assert audit.normalized_case_family_overlap == 1
    assert audit.cited_target_overlap == 1


def test_case_family_normalization_removes_neutral_citation() -> None:
    assert normalize_case_family("Alpha v Beta [2024] SGCA 12") == "alpha v beta"
