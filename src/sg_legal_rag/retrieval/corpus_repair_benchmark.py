from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import (
    DEFAULT_DATA_DIR,
    DEFAULT_SPLITS,
    QueryRecord,
    aggregate_observations,
    load_full_corpus_and_queries,
)
from .bm25 import BM25Index
from .checkpoint import BenchmarkCheckpoint
from .corpus_repair import (
    CorpusRepairDataset,
    load_corpus_repair_dataset,
    max_aggregate_passage_scores,
    max_aggregate_sparse_passage_scores,
)
from .dense import ranking_from_scores
from .dense_benchmark import (
    DEFAULT_CONFIG as DEFAULT_DENSE_CONFIG,
)
from .dense_benchmark import (
    DEFAULT_EMBEDDING_CACHE,
    DEFAULT_MODEL_CACHE,
    DenseModelConfig,
    load_model,
    query_embedding_lookup,
    unique_prepared_texts,
)
from .dense_benchmark import (
    load_config as load_dense_config,
)
from .embedding import artifact_metadata, encode_with_cache, load_cached_embeddings, texts_digest
from .metrics import mean_metrics, metrics_for_ranking

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "corpus_repair.toml"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"


@dataclass(frozen=True)
class CorpusRepairConfig:
    evidence_cutoff_year: int
    max_passage_chars: int
    max_profile_passages: int
    max_profile_identifier_chars: int
    max_profile_context_chars: int
    max_profile_chars: int
    k1: float
    b: float
    dense_model_key: str
    k_values: tuple[int, ...]
    score_batch_size: int


def load_config(path: Path) -> CorpusRepairConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = CorpusRepairConfig(
        evidence_cutoff_year=int(raw["corpus"]["evidence_cutoff_year"]),
        max_passage_chars=int(raw["corpus"]["max_passage_chars"]),
        max_profile_passages=int(raw["corpus"]["max_profile_passages"]),
        max_profile_identifier_chars=int(raw["corpus"]["max_profile_identifier_chars"]),
        max_profile_context_chars=int(raw["corpus"]["max_profile_context_chars"]),
        max_profile_chars=int(raw["corpus"]["max_profile_chars"]),
        k1=float(raw["bm25"]["k1"]),
        b=float(raw["bm25"]["b"]),
        dense_model_key=str(raw["dense"]["model_key"]),
        k_values=tuple(int(value) for value in raw["evaluation"]["k_values"]),
        score_batch_size=int(raw["evaluation"]["score_batch_size"]),
    )
    positive_values = (
        config.evidence_cutoff_year,
        config.max_passage_chars,
        config.max_profile_passages,
        config.max_profile_identifier_chars,
        config.max_profile_context_chars,
        config.max_profile_chars,
        config.k1,
        config.score_batch_size,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("corpus-repair limits and BM25 k1 must be positive")
    if not 0 <= config.b <= 1:
        raise ValueError("BM25 b must be between zero and one")
    if not config.k_values or any(value < 1 for value in config.k_values):
        raise ValueError("evaluation k values must be non-empty and positive")
    return config


def representation_documents(
    dataset: CorpusRepairDataset, representation: str
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    if representation == "identifier":
        documents = dataset.case_texts
        case_ids = np.arange(len(dataset.case_keys), dtype=np.int64)
        policy = {
            "name": "identifier_only_canonical_control",
            "document_source_fields": ["Cited Case"],
            "case_aggregation": "one document per conservative canonical case ID",
            "cold_start_fallback": "not applicable",
        }
    elif representation == "passages":
        documents = tuple(context.text for context in dataset.contexts)
        case_ids = dataset.context_case_ids
        policy = {
            "name": "historical_citation_passages",
            "document_source_fields": [
                "Paragraph",
                "Cited Case",
                "Year",
                "Judgment_URL",
                "Judgment_Reference",
            ],
            "retrieved_text_field": "Paragraph citation-centred window",
            "case_aggregation": "maximum passage score",
            "cold_start_fallback": "none; cases without historical evidence are absent",
        }
    elif representation == "profile":
        documents = dataset.profiles
        case_ids = np.arange(len(dataset.case_keys), dtype=np.int64)
        policy = {
            "name": "historical_case_profile",
            "document_source_fields": [
                "Cited Case",
                "Paragraph",
                "Year",
                "Judgment_URL",
                "Judgment_Reference",
            ],
            "case_aggregation": "one bounded profile per conservative canonical case ID",
            "cold_start_fallback": "case identifier only",
        }
    else:
        raise ValueError(f"unknown candidate representation: {representation}")
    if not documents:
        raise ValueError(f"{representation} produced no candidate documents")
    return documents, case_ids, policy


def selected_queries(queries: list[QueryRecord], max_queries: int | None) -> list[QueryRecord]:
    return queries if max_queries is None else queries[:max_queries]


def relevant_ids(query: QueryRecord, case_to_id: dict[str, int]) -> set[int]:
    return {case_to_id[text] for text in query.relevant_texts}


def benchmark_signature(
    *,
    representation: str,
    retriever: str,
    config: CorpusRepairConfig,
    documents: tuple[str, ...],
    document_case_ids: np.ndarray,
    queries_by_mode: dict[str, list[QueryRecord]],
    max_queries: int | None,
    dense_config: DenseModelConfig | None,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "checkpoint_schema": 1,
        "representation": representation,
        "retriever": retriever,
        "config": asdict(config),
        "max_queries": max_queries,
        "dense_model": asdict(dense_config) if dense_config is not None else None,
        "documents_digest": texts_digest(documents),
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    digest.update(document_case_ids.tobytes())
    for mode, all_queries in queries_by_mode.items():
        digest.update(mode.encode("utf-8"))
        for query in selected_queries(all_queries, max_queries):
            query_payload = (
                query.text,
                sorted(query.relevant_texts),
                sorted(query.courts),
                sorted(query.years),
            )
            encoded = json.dumps(query_payload, ensure_ascii=False).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _breakdowns(
    observations_by_court: dict[str, list[dict[str, float]]],
    observations_by_year: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    return {
        "court": {
            name: {"queries": len(items), "metrics": mean_metrics(items)}
            for name, items in sorted(observations_by_court.items())
        },
        "year": {
            name: {"queries": len(items), "metrics": mean_metrics(items)}
            for name, items in sorted(observations_by_year.items())
        },
    }


def _record_scores(
    state: dict[str, Any],
    query: QueryRecord,
    scores: np.ndarray,
    score_latency_ms: float,
    candidate_case_ids: np.ndarray,
    case_to_id: dict[str, int],
    historical_case_ids: frozenset[int],
    k_values: tuple[int, ...],
) -> None:
    all_relevant = relevant_ids(query, case_to_id)
    rank_started = time.perf_counter()
    ranking = ranking_from_scores(
        scores,
        candidate_case_ids,
        all_relevant,
        top_k=max(k_values),
        candidate_ids_validated=True,
        allow_missing_relevant=True,
    )
    latency_ms = score_latency_ms + (time.perf_counter() - rank_started) * 1000
    state["observations"].append(metrics_for_ranking(ranking, all_relevant, k_values))
    state["latencies_ms"].append(latency_ms)
    state["relevant_counts"].append(len(all_relevant))

    warm_relevant = all_relevant & historical_case_ids
    if warm_relevant:
        warm_ranking = ranking_from_scores(
            scores,
            candidate_case_ids,
            warm_relevant,
            top_k=max(k_values),
            candidate_ids_validated=True,
            allow_missing_relevant=True,
        )
        state["warm_observations"].append(
            metrics_for_ranking(warm_ranking, warm_relevant, k_values)
        )
        state["warm_latencies_ms"].append(latency_ms)
        state["warm_relevant_counts"].append(len(warm_relevant))
    state["queries_completed"] += 1


def _finalize_mode(queries: list[QueryRecord], state: dict[str, Any]) -> dict[str, Any]:
    if state["queries_completed"] != len(queries):
        raise ValueError("cannot finalize an incomplete benchmark mode")
    by_court: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_year: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query, observation in zip(queries, state["observations"], strict=True):
        by_court[query.court].append(observation)
        by_year[query.year].append(observation)
    result = {
        "all_queries": aggregate_observations(
            state["observations"],
            state["latencies_ms"],
            state["relevant_counts"],
        ),
        "warm_start_queries": aggregate_observations(
            state["warm_observations"],
            state["warm_latencies_ms"],
            state["warm_relevant_counts"],
        ),
    }
    result["all_queries"]["breakdowns"] = _breakdowns(by_court, by_year)
    result["warm_start_definition"] = (
        "query has at least one labelled target with <= cutoff historical context; warm metrics "
        "use only retrievable warm targets in the relevance denominator"
    )
    return result


def evaluate_bm25_mode(
    mode: str,
    index: BM25Index,
    document_case_ids: np.ndarray,
    queries: list[QueryRecord],
    case_to_id: dict[str, int],
    historical_case_ids: frozenset[int],
    k_values: tuple[int, ...],
    checkpoint: BenchmarkCheckpoint,
) -> dict[str, Any]:
    candidate_case_ids = np.unique(document_case_ids)
    passage_case_positions = np.searchsorted(candidate_case_ids, document_case_ids)
    if not np.array_equal(candidate_case_ids[passage_case_positions], document_case_ids):
        raise ValueError("passage-to-case mapping is inconsistent")
    state = checkpoint.state_for(mode)
    completed = int(state["queries_completed"])
    if completed > len(queries):
        raise ValueError(f"checkpoint for {mode} exceeds selected query count")
    if completed:
        print(f"resuming {mode}: {completed}/{len(queries)}", flush=True)
    for query in queries[completed:]:
        started = time.perf_counter()
        passage_scores = index.scores(query.text)
        case_scores = max_aggregate_sparse_passage_scores(
            passage_scores,
            passage_case_positions,
            len(candidate_case_ids),
        )
        _record_scores(
            state,
            query,
            case_scores,
            (time.perf_counter() - started) * 1000,
            candidate_case_ids,
            case_to_id,
            historical_case_ids,
            k_values,
        )
        checkpoint.save_progress(mode, len(queries))
    checkpoint.save_progress(mode, len(queries), force=True)
    return _finalize_mode(queries, state)


def run_bm25(
    dataset: CorpusRepairDataset,
    documents: tuple[str, ...],
    document_case_ids: np.ndarray,
    config: CorpusRepairConfig,
    *,
    max_queries: int | None,
    checkpoint: BenchmarkCheckpoint,
) -> dict[str, Any]:
    started = time.perf_counter()
    index = BM25Index(documents, k1=config.k1, b=config.b)
    index_build_ms = (time.perf_counter() - started) * 1000
    case_to_id = {key: index for index, key in enumerate(dataset.case_keys)}
    modes: dict[str, Any] = {}
    for mode, all_queries in dataset.queries_by_mode.items():
        modes[mode] = evaluate_bm25_mode(
            mode,
            index,
            document_case_ids,
            selected_queries(all_queries, max_queries),
            case_to_id,
            dataset.historical_case_ids,
            config.k_values,
            checkpoint,
        )
    return {
        "retriever": "bm25",
        "index_build_ms": index_build_ms,
        "modes": modes,
    }


def _evaluate_dense_mode(
    mode: str,
    queries: list[QueryRecord],
    document_embeddings: np.ndarray,
    document_case_ids: np.ndarray,
    query_embeddings: np.ndarray,
    query_indices: list[int],
    score_batch_size: int,
    case_to_id: dict[str, int],
    historical_case_ids: frozenset[int],
    k_values: tuple[int, ...],
    checkpoint: BenchmarkCheckpoint,
) -> dict[str, Any]:
    candidate_case_ids = np.unique(document_case_ids)
    one_document_per_case = len(candidate_case_ids) == len(document_case_ids)
    document_transposed = document_embeddings.T
    state = checkpoint.state_for(mode)
    completed = int(state["queries_completed"])
    if completed > len(queries):
        raise ValueError(f"checkpoint for {mode} exceeds selected query count")
    if completed:
        print(f"resuming {mode}: {completed}/{len(queries)}", flush=True)
    for offset in range(completed, len(query_indices), score_batch_size):
        batch_indices = query_indices[offset : offset + score_batch_size]
        started = time.perf_counter()
        batch_scores = np.asarray(
            query_embeddings[batch_indices] @ document_transposed,
            dtype=np.float32,
        )
        score_ms = (time.perf_counter() - started) * 1000 / len(batch_indices)
        for query, passage_scores in zip(
            queries[offset : offset + len(batch_indices)], batch_scores, strict=True
        ):
            aggregate_started = time.perf_counter()
            if one_document_per_case:
                case_scores = passage_scores
            else:
                aggregated_ids, case_scores = max_aggregate_passage_scores(
                    passage_scores, document_case_ids
                )
                if not np.array_equal(aggregated_ids, candidate_case_ids):
                    raise ValueError("dense passage aggregation changed candidate case order")
            latency_ms = score_ms + (time.perf_counter() - aggregate_started) * 1000
            _record_scores(
                state,
                query,
                case_scores,
                latency_ms,
                candidate_case_ids,
                case_to_id,
                historical_case_ids,
                k_values,
            )
            checkpoint.save_progress(mode, len(queries))
    checkpoint.save_progress(mode, len(queries), force=True)
    return _finalize_mode(queries, state)


def run_dense(
    dataset: CorpusRepairDataset,
    documents: tuple[str, ...],
    document_case_ids: np.ndarray,
    representation: str,
    config: CorpusRepairConfig,
    dense_config: DenseModelConfig,
    model: Any,
    *,
    data_dir: Path,
    split_path: Path,
    embedding_cache: Path,
    max_queries: int | None,
    checkpoint: BenchmarkCheckpoint,
) -> dict[str, Any]:
    queries_by_mode = {
        mode: selected_queries(queries, max_queries)
        for mode, queries in dataset.queries_by_mode.items()
    }
    prepared_queries = unique_prepared_texts(
        [query.text for queries in queries_by_mode.values() for query in queries],
        dense_config.query_prefix,
    )
    reusable_documents = None
    reusable_document_embeddings = None
    if representation == "identifier":
        baseline_documents, _ = load_full_corpus_and_queries(
            data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv", split_path
        )
        baseline_artifact = load_cached_embeddings(
            baseline_documents,
            cache_dir=embedding_cache,
            model_key=dense_config.key,
            revision=dense_config.revision,
            role="full_corpus",
            dimensions=dense_config.dimensions,
        )
        if baseline_artifact is not None:
            reusable_documents = baseline_documents
            reusable_document_embeddings = baseline_artifact.values

    reusable_queries_artifact = load_cached_embeddings(
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="full_queries",
        dimensions=dense_config.dimensions,
    )
    document_artifact = encode_with_cache(
        model,
        documents,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role=f"corpus_repair_{representation}_documents",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
        reuse_texts=reusable_documents,
        reuse_values=reusable_document_embeddings,
    )
    query_artifact = encode_with_cache(
        model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="corpus_repair_queries",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
        reuse_texts=prepared_queries if reusable_queries_artifact is not None else None,
        reuse_values=(
            reusable_queries_artifact.values if reusable_queries_artifact is not None else None
        ),
    )
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    case_to_id = {key: index for index, key in enumerate(dataset.case_keys)}
    modes: dict[str, Any] = {}
    for mode, queries in queries_by_mode.items():
        query_indices = [
            query_to_index[f"{dense_config.query_prefix}{query.text}"] for query in queries
        ]
        modes[mode] = _evaluate_dense_mode(
            mode,
            queries,
            document_artifact.values,
            document_case_ids,
            query_artifact.values,
            query_indices,
            config.score_batch_size,
            case_to_id,
            dataset.historical_case_ids,
            config.k_values,
            checkpoint,
        )
    return {
        "retriever": "bge_small",
        "model": asdict(dense_config),
        "model_max_sequence_length": model.max_seq_length,
        "encoding": {
            "documents": artifact_metadata(document_artifact),
            "queries": artifact_metadata(query_artifact),
            "unique_query_texts": len(prepared_queries),
        },
        "modes": modes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark leakage-safe SG-LegalCite historical candidate representations"
    )
    parser.add_argument(
        "--representation", choices=("identifier", "passages", "profile"), required=True
    )
    parser.add_argument("--retriever", choices=("bm25", "bge"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="restart-safe progress file; defaults under data/processed/checkpoints",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
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
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be positive")
        config = load_config(args.config)
        dataset = load_corpus_repair_dataset(
            args.data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv",
            args.splits,
            evidence_cutoff_year=config.evidence_cutoff_year,
            max_passage_chars=config.max_passage_chars,
            max_profile_passages=config.max_profile_passages,
            max_profile_identifier_chars=config.max_profile_identifier_chars,
            max_profile_context_chars=config.max_profile_context_chars,
            max_profile_chars=config.max_profile_chars,
        )
        documents, document_case_ids, representation_policy = representation_documents(
            dataset, args.representation
        )
        dense_config = None
        if args.retriever == "bge":
            dense_config, _, _ = load_dense_config(args.dense_config, config.dense_model_key)
        signature = benchmark_signature(
            representation=args.representation,
            retriever=args.retriever,
            config=config,
            documents=documents,
            document_case_ids=document_case_ids,
            queries_by_mode=dataset.queries_by_mode,
            max_queries=args.max_queries,
            dense_config=dense_config,
        )
        checkpoint_path = args.checkpoint or (
            DEFAULT_CHECKPOINT_DIR / f"{args.representation}_{args.retriever}.json"
        )
        checkpoint = BenchmarkCheckpoint.load(
            checkpoint_path,
            signature,
            args.checkpoint_every,
        )
        resumed_queries = {
            mode: int(state["queries_completed"])
            for mode, state in sorted(checkpoint.modes.items())
            if state["queries_completed"]
        }
        model_load_ms = None
        if args.retriever == "bm25":
            benchmark = run_bm25(
                dataset,
                documents,
                document_case_ids,
                config,
                max_queries=args.max_queries,
                checkpoint=checkpoint,
            )
        else:
            if dense_config is None:
                raise ValueError("dense configuration was not loaded")
            model, model_load_ms = load_model(dense_config, args.model_cache)
            benchmark = run_dense(
                dataset,
                documents,
                document_case_ids,
                args.representation,
                config,
                dense_config,
                model,
                data_dir=args.data_dir,
                split_path=args.splits,
                embedding_cache=args.embedding_cache,
                max_queries=args.max_queries,
                checkpoint=checkpoint,
            )
    except Exception as error:  # noqa: BLE001
        print(f"corpus-repair benchmark failed: {error}", file=sys.stderr)
        return 1

    result = {
        "protocol": "full_corpus_temporal_test_historical_context",
        "representation": {
            **representation_policy,
            "documents": len(documents),
            "candidate_cases": len(np.unique(document_case_ids)),
        },
        "dataset_audit": dataset.audit,
        "configuration": {
            **asdict(config),
            "max_queries": args.max_queries,
            "identity_policy": "NFKC, whitespace collapse, and case-folding only",
            "context_order": "source year descending, then URL, reference, and digest ascending",
            "context_deduplication": "SHA-256 of NFKC whitespace-collapsed Paragraph per case",
            "passage_window": "citation-centred when the cited identifier occurs; otherwise prefix",
            "model_load_ms": model_load_ms,
            "tie_break": "ascending conservative canonical case ID",
            "checkpoint_every_queries": args.checkpoint_every,
            "resumed_queries_by_mode": resumed_queries,
        },
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint.clear()
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
