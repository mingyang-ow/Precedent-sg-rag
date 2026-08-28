from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sg_legal_rag.ingestion.validation import EXPECTED_FIELDS, REQUIRED_TEXT_FIELDS

from .bm25 import BM25Index
from .metrics import mean_metrics, metrics_for_ranking

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits_temporal.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bm25.toml"


@dataclass
class QueryRecord:
    text: str
    relevant_texts: set[str] = field(default_factory=set)
    courts: set[str] = field(default_factory=set)
    years: set[str] = field(default_factory=set)

    @property
    def court(self) -> str:
        return next(iter(self.courts)) if len(self.courts) == 1 else "MIXED"

    @property
    def year(self) -> str:
        return next(iter(self.years)) if len(self.years) == 1 else "MIXED"


def percentile(values: Iterable[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_config(path: Path) -> tuple[float, float, tuple[int, ...]]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    return (
        float(raw["bm25"]["k1"]),
        float(raw["bm25"]["b"]),
        tuple(int(value) for value in raw["evaluation"]["k_values"]),
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_lookup_corpus(data_dir: Path) -> list[str]:
    lookup = load_json(data_dir / "stage2_case_lookup.json")
    identifiers = sorted(int(identifier) for identifier in lookup)
    if identifiers != list(range(len(identifiers))):
        raise ValueError("case lookup IDs must be contiguous from zero")
    return [lookup[str(identifier)] for identifier in identifiers]


def aggregate_observations(
    observations: list[dict[str, float]],
    latencies_ms: list[float],
    relevant_counts: list[int],
) -> dict[str, Any]:
    return {
        "queries": len(observations),
        "metrics": mean_metrics(observations),
        "relevant_documents_per_query": {
            "mean": statistics.fmean(relevant_counts),
            "p50": percentile(relevant_counts, 0.50),
            "p95": percentile(relevant_counts, 0.95),
            "max": max(relevant_counts),
        },
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
    }


def evaluate_pools(
    index: BM25Index,
    pools: dict[str, dict[str, Any]],
    query_field: str,
    k_values: tuple[int, ...],
    max_queries: int | None,
) -> dict[str, Any]:
    observations: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    pool_items = sorted(pools.items(), key=lambda item: int(item[0]))
    if max_queries is not None:
        pool_items = pool_items[:max_queries]
    for _, item in pool_items:
        started = time.perf_counter()
        correct_id = int(item["correct_case_id"])
        ranking = index.rank(
            str(item[query_field]),
            {correct_id},
            top_k=max(k_values),
            candidate_indices=[int(identifier) for identifier in item["pool"]],
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        observations.append(metrics_for_ranking(ranking, {correct_id}, k_values))
    return aggregate_observations(observations, latencies_ms, [1] * len(observations))


def run_pooled(
    data_dir: Path,
    *,
    k1: float,
    b: float,
    k_values: tuple[int, ...],
    max_queries: int | None,
) -> dict[str, Any]:
    corpus = build_lookup_corpus(data_dir)
    started = time.perf_counter()
    index = BM25Index(corpus, k1=k1, b=b)
    index_build_ms = (time.perf_counter() - started) * 1000
    direct = load_json(data_dir / "stage2_direct_candidate_pools_v2.json")
    principle = load_json(data_dir / "stage2_single_stage_pools.json")
    modes = {
        "authors_fact_pool": (direct, "fact_text"),
        "paired_fact_only": (principle, "fact_text"),
        "paired_principle_only": (principle, "principle_text"),
        "paired_facts_principle": (principle, "query_text"),
    }
    return {
        "protocol": "pooled",
        "candidate_corpus": "upstream stage2_case_lookup.json raw Cited Case strings",
        "corpus_size": len(corpus),
        "index_build_ms": index_build_ms,
        "modes": {
            mode: evaluate_pools(index, pools, field_name, k_values, max_queries)
            for mode, (pools, field_name) in modes.items()
        },
    }


def load_test_urls(split_path: Path) -> set[str]:
    with split_path.open("r", encoding="utf-8", newline="") as stream:
        return {row["Judgment_URL"] for row in csv.DictReader(stream) if row["Split"] == "test"}


def add_query(
    builders: dict[str, QueryRecord],
    key: str,
    text: str,
    relevant_text: str,
    court: str,
    year: str,
) -> None:
    record = builders.setdefault(key, QueryRecord(text=text))
    record.relevant_texts.add(relevant_text)
    record.courts.add(court)
    record.years.add(year)


def load_full_corpus_and_queries(
    csv_path: Path, split_path: Path
) -> tuple[list[str], dict[str, list[QueryRecord]]]:
    test_urls = load_test_urls(split_path)
    candidates: set[str] = set()
    builders: dict[str, dict[str, QueryRecord]] = {
        "facts_only": {},
        "principle_only": {},
        "facts_principle": {},
    }
    csv.field_size_limit(sys.maxsize)
    with csv_path.open("r", encoding="latin-1", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError("core CSV schema mismatch")
        for row in reader:
            if any(not (row.get(field) or "").strip() for field in REQUIRED_TEXT_FIELDS):
                continue
            cited_case = row["Cited Case"].strip()
            candidates.add(cited_case)
            if row["Judgment_URL"].strip() not in test_urls:
                continue
            fact = row["Fact_Query"].strip()
            principle = row["Key Principles Illustrated"].strip()
            court = row["Court_Type"].strip()
            year = row["Year"].strip()
            add_query(builders["facts_only"], fact, fact, cited_case, court, year)
            add_query(
                builders["principle_only"],
                principle,
                principle,
                cited_case,
                court,
                year,
            )
            add_query(
                builders["facts_principle"],
                f"{fact}\0{principle}",
                f"{fact} {principle}",
                cited_case,
                court,
                year,
            )
    return sorted(candidates), {
        mode: [records[key] for key in sorted(records)] for mode, records in builders.items()
    }


def evaluate_full_queries(
    index: BM25Index,
    candidate_to_index: dict[str, int],
    queries: list[QueryRecord],
    k_values: tuple[int, ...],
    max_queries: int | None,
) -> dict[str, Any]:
    if max_queries is not None:
        queries = queries[:max_queries]
    observations: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    relevant_counts: list[int] = []
    by_court: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_year: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query in queries:
        relevant_indices = {candidate_to_index[text] for text in query.relevant_texts}
        relevant_counts.append(len(relevant_indices))
        started = time.perf_counter()
        ranking = index.rank(query.text, relevant_indices, top_k=max(k_values))
        latencies_ms.append((time.perf_counter() - started) * 1000)
        observation = metrics_for_ranking(ranking, relevant_indices, k_values)
        observations.append(observation)
        by_court[query.court].append(observation)
        by_year[query.year].append(observation)
    result = aggregate_observations(observations, latencies_ms, relevant_counts)
    result["breakdowns"] = {
        "court": {
            name: {"queries": len(items), "metrics": mean_metrics(items)}
            for name, items in sorted(by_court.items())
        },
        "year": {
            name: {"queries": len(items), "metrics": mean_metrics(items)}
            for name, items in sorted(by_year.items())
        },
    }
    return result


def run_full(
    data_dir: Path,
    split_path: Path,
    *,
    k1: float,
    b: float,
    k_values: tuple[int, ...],
    max_queries: int | None,
) -> dict[str, Any]:
    corpus, queries_by_mode = load_full_corpus_and_queries(
        data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv", split_path
    )
    started = time.perf_counter()
    index = BM25Index(corpus, k1=k1, b=b)
    index_build_ms = (time.perf_counter() - started) * 1000
    candidate_to_index = {text: index for index, text in enumerate(corpus)}
    return {
        "protocol": "full_corpus_temporal_test",
        "candidate_corpus": "all unique eligible raw Cited Case strings",
        "corpus_size": len(corpus),
        "index_build_ms": index_build_ms,
        "modes": {
            mode: evaluate_full_queries(index, candidate_to_index, queries, k_values, max_queries)
            for mode, queries in queries_by_mode.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible SG-LegalCite BM25 baselines")
    parser.add_argument("--protocol", choices=("pooled", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="deterministic smoke-test limit; never use for final metrics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.max_queries is not None and args.max_queries < 1:
            raise ValueError("--max-queries must be positive")
        k1, b, k_values = load_config(args.config)
        if args.protocol == "pooled":
            result = run_pooled(
                args.data_dir,
                k1=k1,
                b=b,
                k_values=k_values,
                max_queries=args.max_queries,
            )
        else:
            result = run_full(
                args.data_dir,
                args.splits,
                k1=k1,
                b=b,
                k_values=k_values,
                max_queries=args.max_queries,
            )
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"BM25 benchmark failed: {error}", file=sys.stderr)
        return 1
    result["configuration"] = {
        "k1": k1,
        "b": b,
        "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
        "tokenizer": "NFKC casefolded Unicode word tokens",
        "tie_break": "ascending candidate ID",
        "k_values": list(k_values),
        "max_queries": args.max_queries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
