from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class PoolReport:
    path: str
    queries: int
    pool_sizes: dict[int, int]
    correct_not_in_pool: int
    pools_with_duplicate_ids: int
    ids_missing_from_lookup: int
    correct_names_mismatching_lookup: int
    empty_query_fields: int
    unique_fact_target_pairs: int


@dataclass(frozen=True)
class BenchmarkReport:
    lookup_entries: int
    direct: PoolReport
    principle_augmented: PoolReport
    shared_query_ids: int
    direct_only_query_ids: int
    principle_only_query_ids: int
    numeric_ids_with_same_fact_target: int
    shared_fact_target_pairs: int
    principle_duplicate_fact_target_pairs: int
    warnings: tuple[str, ...]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_pool(
    path: Path,
    lookup: dict[str, str],
    required_query_fields: tuple[str, ...],
) -> tuple[PoolReport, dict[str, tuple[str, int]]]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"{path.name}: expected a JSON object")

    pool_sizes = Counter[int]()
    correct_not_in_pool = 0
    pools_with_duplicates = 0
    missing_lookup_ids: set[int] = set()
    name_mismatches = 0
    empty_query_fields = 0
    semantic_pairs: dict[str, tuple[str, int]] = {}

    for query_id, item in raw.items():
        if not isinstance(item, dict):
            raise TypeError(f"{path.name}:{query_id}: expected an object")
        pool = item.get("pool")
        correct_id = item.get("correct_case_id")
        if not isinstance(pool, list) or not all(isinstance(case_id, int) for case_id in pool):
            raise TypeError(f"{path.name}:{query_id}: pool must be a list of integer IDs")
        if not isinstance(correct_id, int):
            raise TypeError(f"{path.name}:{query_id}: correct_case_id must be an integer")

        pool_sizes[len(pool)] += 1
        if item.get("pool_size") != len(pool):
            raise ValueError(f"{path.name}:{query_id}: pool_size does not match pool length")
        if correct_id not in pool:
            correct_not_in_pool += 1
        if len(pool) != len(set(pool)):
            pools_with_duplicates += 1
        missing_lookup_ids.update(case_id for case_id in pool if str(case_id) not in lookup)

        lookup_name = lookup.get(str(correct_id))
        if lookup_name is not None and item.get("correct_case_name") != lookup_name:
            name_mismatches += 1
        if any(not str(item.get(field, "")).strip() for field in required_query_fields):
            empty_query_fields += 1
        semantic_pairs[query_id] = (str(item.get("fact_text", "")), correct_id)

    return (
        PoolReport(
            path=str(path),
            queries=len(raw),
            pool_sizes=dict(sorted(pool_sizes.items())),
            correct_not_in_pool=correct_not_in_pool,
            pools_with_duplicate_ids=pools_with_duplicates,
            ids_missing_from_lookup=len(missing_lookup_ids),
            correct_names_mismatching_lookup=name_mismatches,
            empty_query_fields=empty_query_fields,
            unique_fact_target_pairs=len(set(semantic_pairs.values())),
        ),
        semantic_pairs,
    )


def inspect_benchmark(data_dir: Path) -> BenchmarkReport:
    lookup_path = data_dir / "stage2_case_lookup.json"
    lookup = _load_json(lookup_path)
    if not isinstance(lookup, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in lookup.items()
    ):
        raise ValueError(f"{lookup_path.name}: expected a string-to-string JSON object")

    direct, direct_pairs_by_id = validate_pool(
        data_dir / "stage2_direct_candidate_pools_v2.json",
        lookup,
        required_query_fields=("fact_text",),
    )
    principle, principle_pairs_by_id = validate_pool(
        data_dir / "stage2_single_stage_pools.json",
        lookup,
        required_query_fields=("fact_text", "principle_text", "query_text"),
    )

    warnings: list[str] = []
    if len(lookup) != 48_478:
        warnings.append(
            f"CSV card reports 48478 unique cited strings, but lookup contains {len(lookup)} IDs"
        )
    direct_ids = set(direct_pairs_by_id)
    principle_ids = set(principle_pairs_by_id)
    direct_pairs = set(direct_pairs_by_id.values())
    principle_pairs = set(principle_pairs_by_id.values())
    numeric_ids_with_same_pair = sum(
        direct_pairs_by_id[query_id] == principle_pairs_by_id[query_id]
        for query_id in direct_ids & principle_ids
    )
    if numeric_ids_with_same_pair != len(direct_ids & principle_ids):
        warnings.append(
            "numeric pool IDs are not semantic pairing keys; paired query-mode comparisons must "
            "rescore the same candidate pool rather than join releases by ID"
        )
    for label, report in (("direct", direct), ("principle_augmented", principle)):
        if report.pool_sizes != {1000: report.queries}:
            warnings.append(f"{label} pools are not uniformly 1000-way")
        if report.correct_not_in_pool:
            warnings.append(f"{label} pools omit the gold ID in some queries")
        if report.ids_missing_from_lookup:
            warnings.append(f"{label} pools contain IDs missing from the case lookup")

    return BenchmarkReport(
        lookup_entries=len(lookup),
        direct=direct,
        principle_augmented=principle,
        shared_query_ids=len(direct_ids & principle_ids),
        direct_only_query_ids=len(direct_ids - principle_ids),
        principle_only_query_ids=len(principle_ids - direct_ids),
        numeric_ids_with_same_fact_target=numeric_ids_with_same_pair,
        shared_fact_target_pairs=len(direct_pairs & principle_pairs),
        principle_duplicate_fact_target_pairs=len(principle_pairs_by_id) - len(principle_pairs),
        warnings=tuple(warnings),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SG-LegalCite benchmark pools")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = inspect_benchmark(args.data_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark validation failed: {error}")
        return 1
    rendered = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
