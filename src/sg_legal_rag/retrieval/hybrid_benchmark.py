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
)
from .bm25 import BM25Index, Ranking
from .dense import ranking_from_scores
from .dense_benchmark import (
    DEFAULT_CONFIG as DEFAULT_DENSE_CONFIG,
)
from .dense_benchmark import (
    DEFAULT_EMBEDDING_CACHE,
    DEFAULT_MODEL_CACHE,
    DenseModelConfig,
    load_model,
    load_pooled_modes,
    prepare_pooled_query_texts,
    query_embedding_lookup,
    selected_queries,
    unique_prepared_texts,
)
from .dense_benchmark import (
    load_config as load_dense_config,
)
from .embedding import artifact_metadata, encode_with_cache, load_cached_embeddings
from .hybrid import weighted_reciprocal_rank_fusion
from .metrics import mean_metrics, metrics_for_ranking

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hybrid.toml"


@dataclass(frozen=True)
class HybridConfig:
    k1: float
    b: float
    dense_model_key: str
    method: str
    rank_constant: int
    component_depth: int
    bm25_weight: float
    dense_weight: float
    k_values: tuple[int, ...]
    score_batch_size: int


def load_config(path: Path) -> HybridConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = HybridConfig(
        k1=float(raw["bm25"]["k1"]),
        b=float(raw["bm25"]["b"]),
        dense_model_key=str(raw["dense"]["model_key"]),
        method=str(raw["fusion"]["method"]),
        rank_constant=int(raw["fusion"]["rank_constant"]),
        component_depth=int(raw["fusion"]["component_depth"]),
        bm25_weight=float(raw["fusion"]["bm25_weight"]),
        dense_weight=float(raw["fusion"]["dense_weight"]),
        k_values=tuple(int(value) for value in raw["evaluation"]["k_values"]),
        score_batch_size=int(raw["evaluation"]["score_batch_size"]),
    )
    if config.method != "weighted_rrf":
        raise ValueError("only weighted_rrf fusion is supported")
    if config.rank_constant < 1 or config.component_depth < 1:
        raise ValueError("rank constant and component depth must be positive")
    if not config.k_values or any(value < 1 for value in config.k_values):
        raise ValueError("k values must be non-empty and positive")
    if config.component_depth < max(config.k_values):
        raise ValueError("component depth must cover the largest evaluation k")
    if config.score_batch_size < 1:
        raise ValueError("score batch size must be positive")
    for weight in (config.bm25_weight, config.dense_weight):
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("fusion weights must be finite and positive")
    return config


def positive_bm25_candidates(ranking: Ranking) -> tuple[int, ...]:
    return ranking.top_indices[: min(ranking.positive_matches, len(ranking.top_indices))]


def fuse_rankings(
    bm25_ranking: Ranking,
    dense_ranking: Ranking,
    relevant_indices: set[int],
    config: HybridConfig,
    *,
    top_k: int | None = None,
) -> Ranking:
    return weighted_reciprocal_rank_fusion(
        (
            (positive_bm25_candidates(bm25_ranking), config.bm25_weight),
            (dense_ranking.top_indices, config.dense_weight),
        ),
        relevant_indices,
        rank_constant=config.rank_constant,
        top_k=max(config.k_values) if top_k is None else top_k,
    )


def evaluate_pools(
    index: BM25Index,
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    pools: dict[str, dict[str, Any]],
    query_field: str,
    query_prefix: str,
    config: HybridConfig,
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
        correct_id = int(item["correct_case_id"])
        relevant_ids = {correct_id}
        query_text = str(item[query_field])
        bm25_ranking = index.rank(
            query_text,
            relevant_ids,
            top_k=min(config.component_depth, len(candidate_ids)),
            candidate_indices=candidate_ids.tolist(),
        )
        prepared_query = f"{query_prefix}{query_text}"
        query_embedding = query_embeddings[query_to_index[prepared_query]]
        dense_scores = np.asarray(
            corpus_embeddings[candidate_ids] @ query_embedding,
            dtype=np.float32,
        )
        dense_ranking = ranking_from_scores(
            dense_scores,
            candidate_ids,
            relevant_ids,
            top_k=min(config.component_depth, len(candidate_ids)),
        )
        hybrid_ranking = fuse_rankings(bm25_ranking, dense_ranking, relevant_ids, config)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        observations.append(metrics_for_ranking(hybrid_ranking, relevant_ids, config.k_values))

    return aggregate_observations(observations, latencies_ms, [1] * len(observations))


def run_pooled(
    data_dir: Path,
    model: Any,
    dense_config: DenseModelConfig,
    config: HybridConfig,
    *,
    embedding_cache: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    corpus = build_lookup_corpus(data_dir)
    started = time.perf_counter()
    index = BM25Index(corpus, k1=config.k1, b=config.b)
    index_build_ms = (time.perf_counter() - started) * 1000
    modes = load_pooled_modes(data_dir)
    prepared_queries = prepare_pooled_query_texts(
        modes,
        dense_config.query_prefix,
        max_queries,
    )
    corpus_artifact = encode_with_cache(
        model,
        corpus,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_corpus",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
    )
    query_artifact = encode_with_cache(
        model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_queries",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
    )
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    return {
        "protocol": "pooled",
        "candidate_corpus": "upstream stage2_case_lookup.json raw Cited Case strings",
        "corpus_size": len(corpus),
        "index_build_ms": index_build_ms,
        "encoding": {
            "corpus": artifact_metadata(corpus_artifact),
            "queries": artifact_metadata(query_artifact),
            "unique_query_texts": len(prepared_queries),
        },
        "modes": {
            mode: evaluate_pools(
                index,
                corpus_artifact.values,
                query_artifact.values,
                query_to_index,
                pools,
                field_name,
                dense_config.query_prefix,
                config,
                max_queries,
            )
            for mode, (pools, field_name) in modes.items()
        },
    }


def evaluate_full_queries(
    index: BM25Index,
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    candidate_to_index: dict[str, int],
    queries: list[QueryRecord],
    query_prefix: str,
    config: HybridConfig,
) -> dict[str, Any]:
    observations: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    relevant_counts: list[int] = []
    by_court: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_year: dict[str, list[dict[str, float]]] = defaultdict(list)
    candidate_ids = np.arange(len(corpus_embeddings), dtype=np.int64)
    corpus_transposed = corpus_embeddings.T

    for offset in range(0, len(queries), config.score_batch_size):
        batch = queries[offset : offset + config.score_batch_size]
        query_indices = [query_to_index[f"{query_prefix}{query.text}"] for query in batch]
        score_started = time.perf_counter()
        score_rows = np.asarray(
            query_embeddings[query_indices] @ corpus_transposed, dtype=np.float32
        )
        score_ms_per_query = (time.perf_counter() - score_started) * 1000 / len(batch)

        for query, dense_scores in zip(batch, score_rows, strict=True):
            started = time.perf_counter()
            relevant_ids = {candidate_to_index[text] for text in query.relevant_texts}
            relevant_counts.append(len(relevant_ids))
            bm25_ranking = index.rank(
                query.text,
                relevant_ids,
                top_k=min(config.component_depth, len(candidate_ids)),
            )
            dense_ranking = ranking_from_scores(
                dense_scores,
                candidate_ids,
                relevant_ids,
                top_k=min(config.component_depth, len(candidate_ids)),
                candidate_ids_validated=True,
            )
            hybrid_ranking = fuse_rankings(bm25_ranking, dense_ranking, relevant_ids, config)
            latencies_ms.append(score_ms_per_query + (time.perf_counter() - started) * 1000)
            observation = metrics_for_ranking(hybrid_ranking, relevant_ids, config.k_values)
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
    dense_config: DenseModelConfig,
    config: HybridConfig,
    *,
    embedding_cache: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    corpus, all_queries = load_full_corpus_and_queries(
        data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv",
        split_path,
    )
    queries_by_mode = {
        mode: selected_queries(queries, max_queries) for mode, queries in all_queries.items()
    }
    prepared_queries = unique_prepared_texts(
        [query.text for queries in queries_by_mode.values() for query in queries],
        dense_config.query_prefix,
    )
    pooled_corpus = build_lookup_corpus(data_dir)
    pooled_modes = load_pooled_modes(data_dir)
    pooled_queries = prepare_pooled_query_texts(
        pooled_modes,
        dense_config.query_prefix,
        None,
    )
    reusable_corpus = load_cached_embeddings(
        pooled_corpus,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_corpus",
        dimensions=dense_config.dimensions,
    )
    reusable_queries = load_cached_embeddings(
        pooled_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_queries",
        dimensions=dense_config.dimensions,
    )
    corpus_artifact = encode_with_cache(
        model,
        corpus,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="full_corpus",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
        reuse_texts=pooled_corpus if reusable_corpus is not None else None,
        reuse_values=reusable_corpus.values if reusable_corpus is not None else None,
    )
    query_artifact = encode_with_cache(
        model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="full_queries",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
        reuse_texts=pooled_queries if reusable_queries is not None else None,
        reuse_values=reusable_queries.values if reusable_queries is not None else None,
    )
    started = time.perf_counter()
    index = BM25Index(corpus, k1=config.k1, b=config.b)
    index_build_ms = (time.perf_counter() - started) * 1000
    candidate_to_index = {text: index for index, text in enumerate(corpus)}
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    return {
        "protocol": "full_corpus_temporal_test",
        "candidate_corpus": "all unique eligible raw Cited Case strings",
        "corpus_size": len(corpus),
        "index_build_ms": index_build_ms,
        "encoding": {
            "corpus": artifact_metadata(corpus_artifact),
            "queries": artifact_metadata(query_artifact),
            "unique_query_texts": len(prepared_queries),
        },
        "modes": {
            mode: evaluate_full_queries(
                index,
                corpus_artifact.values,
                query_artifact.values,
                query_to_index,
                candidate_to_index,
                queries,
                dense_config.query_prefix,
                config,
            )
            for mode, queries in queries_by_mode.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SG-LegalCite BM25+BGE hybrid baseline")
    parser.add_argument("--protocol", choices=("pooled", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG)
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
        config = load_config(args.config)
        dense_config, _, _ = load_dense_config(args.dense_config, config.dense_model_key)
        model, model_load_ms = load_model(dense_config, args.model_cache)
        if args.protocol == "pooled":
            result = run_pooled(
                args.data_dir,
                model,
                dense_config,
                config,
                embedding_cache=args.embedding_cache,
                max_queries=args.max_queries,
            )
        else:
            result = run_full(
                args.data_dir,
                args.splits,
                model,
                dense_config,
                config,
                embedding_cache=args.embedding_cache,
                max_queries=args.max_queries,
            )
    # This CLI boundary also reports third-party model/download exceptions without a traceback.
    except Exception as error:  # noqa: BLE001
        print(f"hybrid benchmark failed: {error}", file=sys.stderr)
        return 1

    result["components"] = {
        "bm25": {"k1": config.k1, "b": config.b},
        "dense": asdict(dense_config),
    }
    result["configuration"] = {
        "method": config.method,
        "rank_constant": config.rank_constant,
        "component_depth": config.component_depth,
        "bm25_weight": config.bm25_weight,
        "dense_weight": config.dense_weight,
        "tie_break": "ascending candidate ID",
        "k_values": list(config.k_values),
        "score_batch_size": config.score_batch_size,
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
