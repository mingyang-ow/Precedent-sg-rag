from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "COMBINED_ALL_CASES_FINAL_V2.csv"
EXPECTED_FIELDS = (
    "Judgment_URL",
    "Judgment_Reference",
    "Year",
    "Court_Type",
    "Case_Number",
    "Case Name",
    "Current Court Level",
    "Fact_Query",
    "Cited Case",
    "Paragraph",
    "Key Principles Illustrated",
    "Issue",
    "Issue Group",
)
REQUIRED_TEXT_FIELDS = (
    "Judgment_URL",
    "Judgment_Reference",
    "Fact_Query",
    "Cited Case",
    "Key Principles Illustrated",
)
NEUTRAL_CITATION_RE = re.compile(r"\[\d{4}\]\s+SG(?:CA|CAI|HC|HCF|HCR)\s+\d+", re.IGNORECASE)


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot safely enter the evaluation pipeline."""


@dataclass(frozen=True)
class ValidationReport:
    path: str
    rows: int
    unique_judgments: int
    unique_principles: int
    unique_cited_cases: int
    unique_issues: int
    unique_issue_groups: int
    year_min: int
    year_max: int
    courts: dict[str, int]
    missing_by_field: dict[str, int]
    rows_eligible_for_all_query_modes: int
    duplicate_semantic_rows: int
    inconsistent_judgment_metadata: int
    reference_url_conflicts: int
    suspicious_cited_case_values: int
    quality_issue_samples: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(row: dict[str, str], field: str) -> str:
    return (row.get(field) or "").strip()


def inspect_csv(path: Path) -> ValidationReport:
    csv.field_size_limit(sys.maxsize)
    missing = Counter[str]()
    courts = Counter[str]()
    judgments: dict[str, tuple[str, str, str, str]] = {}
    reference_to_url: dict[str, str] = {}
    principles: set[str] = set()
    cited_cases: set[str] = set()
    issues: set[str] = set()
    issue_groups: set[str] = set()
    semantic_keys: set[tuple[str, str, str]] = set()
    years: list[int] = []
    duplicate_rows = 0
    inconsistent_metadata = 0
    reference_url_conflicts = 0
    suspicious_citations = 0
    ineligible_rows = 0
    quality_issue_samples: list[dict[str, Any]] = []
    rows = 0

    try:
        stream = path.open("r", encoding="latin-1", newline="")
    except OSError as error:
        raise DatasetValidationError(f"cannot open dataset: {error}") from error

    with stream:
        reader = csv.DictReader(stream)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != EXPECTED_FIELDS:
            raise DatasetValidationError(
                f"schema mismatch: expected {EXPECTED_FIELDS!r}, found {actual_fields!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            rows += 1
            missing_required_on_row: list[str] = []
            for field in EXPECTED_FIELDS:
                if not _text(row, field):
                    missing[field] += 1
                    if field in REQUIRED_TEXT_FIELDS:
                        missing_required_on_row.append(field)
            if missing_required_on_row:
                ineligible_rows += 1
                if len(quality_issue_samples) < 20:
                    quality_issue_samples.append(
                        {
                            "line": line_number,
                            "judgment_url": _text(row, "Judgment_URL"),
                            "judgment_reference": _text(row, "Judgment_Reference"),
                            "missing_required_fields": missing_required_on_row,
                        }
                    )

            year_text = _text(row, "Year")
            try:
                year = int(year_text)
            except ValueError as error:
                raise DatasetValidationError(
                    f"line {line_number}: invalid Year value {year_text!r}"
                ) from error
            years.append(year)
            courts[_text(row, "Court_Type")] += 1

            url = _text(row, "Judgment_URL")
            reference = _text(row, "Judgment_Reference")
            judgment_metadata = (
                reference,
                year_text,
                _text(row, "Case Name"),
                _text(row, "Fact_Query"),
            )
            previous_metadata = judgments.setdefault(url, judgment_metadata)
            if previous_metadata != judgment_metadata:
                inconsistent_metadata += 1

            previous_url = reference_to_url.setdefault(reference, url)
            if previous_url != url:
                reference_url_conflicts += 1

            principle = _text(row, "Key Principles Illustrated")
            cited_case = _text(row, "Cited Case")
            issue = _text(row, "Issue")
            issue_group = _text(row, "Issue Group")
            principles.add(principle)
            cited_cases.add(cited_case)
            issues.add(issue)
            issue_groups.add(issue_group)

            semantic_key = (url, cited_case, principle)
            if semantic_key in semantic_keys:
                duplicate_rows += 1
            else:
                semantic_keys.add(semantic_key)

            if len(cited_case) > 500 or (
                len(cited_case) > 180 and not NEUTRAL_CITATION_RE.search(cited_case)
            ):
                suspicious_citations += 1

    if not rows:
        raise DatasetValidationError("dataset has no records")
    warnings: list[str] = []
    if rows != 100_890:
        warnings.append(f"upstream card reports 100890 rows; observed {rows}")
    if len(judgments) != 8_523:
        warnings.append(f"upstream card reports 8523 judgments; observed {len(judgments)}")
    if suspicious_citations:
        warnings.append(
            "some Cited Case values resemble extracted passages rather than canonical case identifiers"
        )
    if duplicate_rows:
        warnings.append("duplicate (judgment, cited case, principle) labels require policy review")
    if ineligible_rows:
        warnings.append(
            f"{ineligible_rows} rows are ineligible for common query-mode evaluation because a "
            "required retrieval field is empty"
        )

    return ValidationReport(
        path=str(path),
        rows=rows,
        unique_judgments=len(judgments),
        unique_principles=len(principles - {""}),
        unique_cited_cases=len(cited_cases - {""}),
        unique_issues=len(issues - {""}),
        unique_issue_groups=len(issue_groups - {""}),
        year_min=min(years),
        year_max=max(years),
        courts=dict(sorted(courts.items())),
        missing_by_field={field: missing[field] for field in EXPECTED_FIELDS},
        rows_eligible_for_all_query_modes=rows - ineligible_rows,
        duplicate_semantic_rows=duplicate_rows,
        inconsistent_judgment_metadata=inconsistent_metadata,
        reference_url_conflicts=reference_url_conflicts,
        suspicious_cited_case_values=suspicious_citations,
        quality_issue_samples=tuple(quality_issue_samples),
        warnings=tuple(warnings),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the SG-LegalCite core CSV")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero status when any row is ineligible for all query modes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = inspect_csv(args.dataset)
    except DatasetValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered)
    if args.strict and report.rows_eligible_for_all_query_modes != report.rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
