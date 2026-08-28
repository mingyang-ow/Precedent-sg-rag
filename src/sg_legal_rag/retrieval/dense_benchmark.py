from __future__ import annotations

import argparse
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
    build_lookup_corpus,
    load_full_corpus_and_queries,
    load_json,
)
from .dense import ranking_from_scores
from .embedding import (
    EmbeddingArtifact,
    artifact_metadata,
    encode_with_cache,
    load_cached_embeddings,
)
from .metrics import mean_metrics, metrics_for_ranking

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dense_models.toml"
DEFAULT_EMBEDDING_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "data" / "models"


@dataclass(frozen=True)
class DenseModelConfig:
    key: str
    model_id: str
    revision: str
    dimensions: int
    encode_batch_size: int
    query_prefix: str
    license: str


def load_config(path: Path, model_key: str) -> tuple[DenseModelConfig, tuple[int, ...], int]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    raw_model = raw["models"][model_key]
    model = DenseModelConfig(
        key=model_key,
        model_id=str(raw_model["model_id"]),
        revision=str(raw_model["revision"]),
        dimensions=int(raw_model["dimensions"]),
        encode_batch_size=int(raw_model["encode_batch_size"]),
        query_prefix=str(raw_model["query_prefix"]),
        license=str(raw_model["license"]),
    )
    k_values = tuple(int(value) for value in raw["evaluation"]["k_values"])
    score_batch_size = int(raw["evaluation"]["score_batch_size"])
    if model.dimensions < 1 or model.encode_batch_size < 1 or score_batch_size < 1:
        raise ValueError("dimensions and batch sizes must be positive")
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("k values must be non-empty and positive")
    return model, k_values, score_batch_size


def load_model(config: DenseModelConfig, model_cache: Path) -> tuple[Any, float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "install the dense dependency extra before running this command"
        ) from error

    started = time.perf_counter()
    model = SentenceTransformer(
        config.model_id,
        revision=config.revision,
        device="cpu",
        cache_folder=str(model_cache),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    actual_dimensions = model.get_embedding_dimension()
    if actual_dimensions != config.dimensions:
        raise ValueError(
            f"model dimension mismatch: configured {config.dimensions}, got {actual_dimensions}"
        )
    return model, elapsed_ms


def selected_queries(queries: list[QueryRecord], max_queries: int | None) -> list[QueryRecord]:
    return queries if max_queries is None else queries[:max_queries]


def unique_prepared_texts(raw_texts: list[str], prefix: str) -> list[str]:
    return sorted({f"{prefix}{text}" for text in raw_texts})


def query_embedding_lookup(
    artifact: EmbeddingArtifact, prepared_texts: list[str]
) -> dict[str, int]:
    if len(artifact.values) != len(prepared_texts):
        raise ValueError("query text and embedding counts differ")
    return {text: index for index, text in enumerate(prepared_texts)}


def evaluate_pools(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    pools: dict[str, dict[str, Any]],
    query_field: str,
    query_prefix: str,
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
        candidate_ids = np.asarray(item["pool"], dtype=np.int64)
        query_text = f"{query_prefix}{item[query_field]}"
        query_embedding = query_embeddings[query_to_index[query_text]]
        scores = np.asarray(corpus_embeddings[candidate_ids] @ query_embedding, dtype=np.float32)
        correct_id = int(item["correct_case_id"])
        ranking = ranking_from_scores(
            scores,
            candidate_ids,
            {correct_id},
            top_k=max(k_values),
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        observations.append(metrics_for_ranking(ranking, {correct_id}, k_values))
    return aggregate_observations(observations, latencies_ms, [1] * len(observations))


def load_pooled_modes(data_dir: Path) -> dict[str, tuple[dict[str, dict[str, Any]], str]]:
    direct = load_json(data_dir / "stage2_direct_candidate_pools_v2.json")
    principle = load_json(data_dir / "stage2_single_stage_pools.json")
    return {
        "authors_fact_pool": (direct, "fact_text"),
        "paired_fact_only": (principle, "fact_text"),
        "paired_principle_only": (principle, "principle_text"),
        "paired_facts_principle": (principle, "query_text"),
    }


def prepare_pooled_query_texts(
    modes: dict[str, tuple[dict[str, dict[str, Any]], str]],
    prefix: str,
    max_queries: int | None,
) -> list[str]:
    raw_query_texts: list[str] = []
    for pools, field_name in modes.values():
        items = sorted(pools.items(), key=lambda item: int(item[0]))
        if max_queries is not None:
            items = items[:max_queries]
        raw_query_texts.extend(str(item[field_name]) for _, item in items)
    return unique_prepared_texts(raw_query_texts, prefix)


def run_pooled(
    data_dir: Path,
    model: Any,
    config: DenseModelConfig,
    *,
    k_values: tuple[int, ...],
    embedding_cache: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    corpus = build_lookup_corpus(data_dir)
    modes = load_pooled_modes(data_dir)
    prepared_queries = prepare_pooled_query_texts(modes, config.query_prefix, max_queries)
    corpus_artifact = encode_with_cache(
        model,
        corpus,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="pooled_corpus",
        batch_size=config.encode_batch_size,
        dimensions=config.dimensions,
    )
    query_artifact = encode_with_cache(
        model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="pooled_queries",
        batch_size=config.encode_batch_size,
        dimensions=config.dimensions,
    )
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    return {
        "protocol": "pooled",
        "candidate_corpus": "upstream stage2_case_lookup.json raw Cited Case strings",
        "corpus_size": len(corpus),
        "encoding": {
            "corpus": artifact_metadata(corpus_artifact),
            "queries": artifact_metadata(query_artifact),
            "unique_query_texts": len(prepared_queries),
        },
        "modes": {
            mode: evaluate_pools(
                corpus_artifact.values,
                query_artifact.values,
                query_to_index,
                pools,
                field_name,
                config.query_prefix,
                k_values,
                max_queries,
            )
            for mode, (pools, field_name) in modes.items()
        },
    }


def evaluate_full_queries(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    candidate_to_index: dict[str, int],
    queries: list[QueryRecord],
    query_prefix: str,
    k_values: tuple[int, ...],
    score_batch_size: int,
) -> dict[str, Any]:
    observations: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    relevant_counts: list[int] = []
    by_court: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_year: dict[str, list[dict[str, float]]] = defaultdict(list)
    candidate_ids = np.arange(len(corpus_embeddings), dtype=np.int64)
    corpus_transposed = corpus_embeddings.T

    for offset in range(0, len(queries), score_batch_size):
        batch = queries[offset : offset + score_batch_size]
        query_indices = [query_to_index[f"{query_prefix}{query.text}"] for query in batch]
        score_started = time.perf_counter()
        scores = np.asarray(query_embeddings[query_indices] @ corpus_transposed, dtype=np.float32)
        score_ms_per_query = (time.perf_counter() - score_started) * 1000 / len(batch)
        for query, query_scores in zip(batch, scores, strict=True):
            relevant_ids = {candidate_to_index[text] for text in query.relevant_texts}
            relevant_counts.append(len(relevant_ids))
            rank_started = time.perf_counter()
            ranking = ranking_from_scores(
                query_scores,
                candidate_ids,
                relevant_ids,
                top_k=max(k_values),
                candidate_ids_validated=True,
            )
            latencies_ms.append(score_ms_per_query + (time.perf_counter() - rank_started) * 1000)
            observation = metrics_for_ranking(ranking, relevant_ids, k_values)
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
    model: Any,
    config: DenseModelConfig,
    *,
    k_values: tuple[int, ...],
    score_batch_size: int,
    embedding_cache: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    corpus, all_queries = load_full_corpus_and_queries(
        data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv", split_path
    )
    queries_by_mode = {
        mode: selected_queries(queries, max_queries) for mode, queries in all_queries.items()
    }
    prepared_queries = unique_prepared_texts(
        [query.text for queries in queries_by_mode.values() for query in queries],
        config.query_prefix,
    )
    pooled_corpus = build_lookup_corpus(data_dir)
    pooled_modes = load_pooled_modes(data_dir)
    pooled_queries = prepare_pooled_query_texts(pooled_modes, config.query_prefix, None)
    reusable_corpus = load_cached_embeddings(
        pooled_corpus,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="pooled_corpus",
        dimensions=config.dimensions,
    )
    reusable_queries = load_cached_embeddings(
        pooled_queries,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="pooled_queries",
        dimensions=config.dimensions,
    )
    corpus_artifact = encode_with_cache(
        model,
        corpus,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="full_corpus",
        batch_size=config.encode_batch_size,
        dimensions=config.dimensions,
        reuse_texts=pooled_corpus if reusable_corpus is not None else None,
        reuse_values=reusable_corpus.values if reusable_corpus is not None else None,
    )
    query_artifact = encode_with_cache(
        model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=config.key,
        revision=config.revision,
        role="full_queries",
        batch_size=config.encode_batch_size,
        dimensions=config.dimensions,
        reuse_texts=pooled_queries if reusable_queries is not None else None,
        reuse_values=reusable_queries.values if reusable_queries is not None else None,
    )
    candidate_to_index = {text: index for index, text in enumerate(corpus)}
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    return {
        "protocol": "full_corpus_temporal_test",
        "candidate_corpus": "all unique eligible raw Cited Case strings",
        "corpus_size": len(corpus),
        "encoding": {
            "corpus": artifact_metadata(corpus_artifact),
            "queries": artifact_metadata(query_artifact),
            "unique_query_texts": len(prepared_queries),
        },
        "modes": {
            mode: evaluate_full_queries(
                corpus_artifact.values,
                query_artifact.values,
                query_to_index,
                candidate_to_index,
                queries,
                config.query_prefix,
                k_values,
                score_batch_size,
            )
            for mode, queries in queries_by_mode.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pinned SG-LegalCite dense baselines")
    parser.add_argument("--protocol", choices=("pooled", "full"), required=True)
    parser.add_argument("--model", choices=("minilm", "bge_small", "mpnet"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
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
        config, k_values, score_batch_size = load_config(args.config, args.model)
        model, model_load_ms = load_model(config, args.model_cache)
        if args.protocol == "pooled":
            result = run_pooled(
                args.data_dir,
                model,
                config,
                k_values=k_values,
                embedding_cache=args.embedding_cache,
                max_queries=args.max_queries,
            )
        else:
            result = run_full(
                args.data_dir,
                args.splits,
                model,
                config,
                k_values=k_values,
                score_batch_size=score_batch_size,
                embedding_cache=args.embedding_cache,
                max_queries=args.max_queries,
            )
    # This CLI boundary also reports third-party model/download exceptions without a traceback.
    except Exception as error:  # noqa: BLE001
        print(f"dense benchmark failed: {error}", file=sys.stderr)
        return 1
    result["model"] = asdict(config)
    result["configuration"] = {
        "device": "cpu",
        "similarity": "cosine via normalized embeddings and dot product",
        "tie_break": "ascending candidate ID",
        "k_values": list(k_values),
        "score_batch_size": score_batch_size,
        "max_queries": args.max_queries,
        "model_load_ms": model_load_ms,
        "model_max_sequence_length": model.max_seq_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
