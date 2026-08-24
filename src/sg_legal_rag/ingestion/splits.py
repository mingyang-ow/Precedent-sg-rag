from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .validation import DEFAULT_DATASET, EXPECTED_FIELDS, DatasetValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "splits_temporal.csv"
NEUTRAL_CITATION_SUFFIX_RE = re.compile(
    r"\s*(?:\[\d{4}\]\s+SG(?:CA|CAI|HC|HCF|HCR)\s+\d+|\[\d{4}\]\s+\d+\s+SLR(?:\(R\))?\s+\d+)\s*$",
    re.IGNORECASE,
)
NON_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Judgment:
    url: str
    reference: str
    year: int
    case_name: str


@dataclass(frozen=True)
class SplitAudit:
    strategy: str
    record_counts: dict[str, int]
    judgment_counts: dict[str, int]
    judgment_url_overlap: int
    normalized_case_family_overlap: int
    cited_target_overlap: int
    notes: tuple[str, ...]


def normalize_case_family(case_name: str) -> str:
    without_citation = NEUTRAL_CITATION_SUFFIX_RE.sub("", case_name.strip().lower())
    return NON_WORD_RE.sub(" ", without_citation).strip()


def load_judgments_and_targets(path: Path) -> tuple[dict[str, Judgment], dict[str, list[str]]]:
    csv.field_size_limit(sys.maxsize)
    judgments: dict[str, Judgment] = {}
    targets: dict[str, list[str]] = defaultdict(list)
    try:
        stream = path.open("r", encoding="latin-1", newline="")
    except OSError as error:
        raise DatasetValidationError(f"cannot open dataset: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise DatasetValidationError("schema mismatch; run sg-legal-validate for details")
        for row in reader:
            url = row["Judgment_URL"].strip()
            judgment = Judgment(
                url=url,
                reference=row["Judgment_Reference"].strip(),
                year=int(row["Year"]),
                case_name=row["Case Name"].strip(),
            )
            previous = judgments.setdefault(url, judgment)
            if previous != judgment:
                raise DatasetValidationError(f"inconsistent metadata for judgment {url}")
            targets[url].append(row["Cited Case"].strip())
    return judgments, targets


def temporal_assignments(judgments: dict[str, Judgment]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for url, judgment in judgments.items():
        if judgment.year <= 2021:
            split = "train"
        elif judgment.year <= 2023:
            split = "validation"
        else:
            split = "test"
        assignments[url] = split
    return assignments


def grouped_random_assignments(judgments: dict[str, Judgment], seed: int = 42) -> dict[str, str]:
    urls = sorted(judgments)
    random.Random(seed).shuffle(urls)
    train_end = round(len(urls) * 0.8)
    validation_end = train_end + round(len(urls) * 0.1)
    return {
        url: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, url in enumerate(urls)
    }


def _cross_split_overlap(values_by_split: dict[str, set[str]]) -> int:
    membership = Counter(value for values in values_by_split.values() for value in values if value)
    return sum(count > 1 for count in membership.values())


def audit_assignments(
    strategy: str,
    judgments: dict[str, Judgment],
    targets: dict[str, list[str]],
    assignments: dict[str, str],
) -> SplitAudit:
    urls_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, set[str]] = defaultdict(set)
    targets_by_split: dict[str, set[str]] = defaultdict(set)
    record_counts = Counter[str]()
    judgment_counts = Counter[str]()
    for url, split in assignments.items():
        urls_by_split[split].add(url)
        families_by_split[split].add(normalize_case_family(judgments[url].case_name))
        targets_by_split[split].update(targets[url])
        record_counts[split] += len(targets[url])
        judgment_counts[split] += 1

    return SplitAudit(
        strategy=strategy,
        record_counts=dict(sorted(record_counts.items())),
        judgment_counts=dict(sorted(judgment_counts.items())),
        judgment_url_overlap=_cross_split_overlap(urls_by_split),
        normalized_case_family_overlap=_cross_split_overlap(families_by_split),
        cited_target_overlap=_cross_split_overlap(targets_by_split),
        notes=(
            "Cited-target overlap is expected for transductive retrieval, where one candidate corpus is shared.",
            "Normalized case-family matching is a heuristic audit, not a reliable proceeding-family identifier.",
        ),
    )


def write_assignments(
    path: Path, judgments: dict[str, Judgment], assignments: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Judgment_URL", "Judgment_Reference", "Year", "Split"))
        for url in sorted(assignments):
            judgment = judgments[url]
            writer.writerow((url, judgment.reference, judgment.year, assignments[url]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leak-audited SG-LegalCite splits")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--strategy", choices=("temporal", "grouped-random"), default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        judgments, targets = load_judgments_and_targets(args.dataset)
        if args.strategy == "temporal":
            assignments = temporal_assignments(judgments)
        else:
            assignments = grouped_random_assignments(judgments, seed=args.seed)
        audit = audit_assignments(args.strategy, judgments, targets, assignments)
        if audit.judgment_url_overlap:
            raise DatasetValidationError("judgment URL leakage detected")
        write_assignments(args.output, judgments, assignments)
    except (DatasetValidationError, OSError, ValueError) as error:
        print(f"split creation failed: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(asdict(audit), indent=2, sort_keys=True)
    audit_output = args.audit_output or args.output.with_suffix(".audit.json")
    audit_output.write_text(rendered + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
