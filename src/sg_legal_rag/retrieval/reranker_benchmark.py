from __future__ import annotations

import argparse
import json
import statistics
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
    build_lookup_corpus,
    load_full_corpus_and_queries,
    percentile,
)
from .bm25 import BM25Index
from .dense import ranking_from_scores
from .dense_benchmark import (
    DEFAULT_CONFIG as DEFAULT_DENSE_CONFIG,
)
from .dense_benchmark import (
    DEFAULT_EMBEDDING_CACHE,
    DEFAULT_MODEL_CACHE,
    DenseModelConfig,
    load_pooled_modes,
    prepare_pooled_query_texts,
    query_embedding_lookup,
    selected_queries,
    unique_prepared_texts,
)
from .dense_benchmark import (
    load_config as load_dense_config,
)
from .dense_benchmark import (
    load_model as load_dense_model,
)
from .embedding import artifact_metadata, encode_with_cache, load_cached_embeddings
from .hybrid_benchmark import (
    DEFAULT_CONFIG as DEFAULT_HYBRID_CONFIG,
)
from .hybrid_benchmark import (
    HybridConfig,
    fuse_rankings,
)
from .hybrid_benchmark import (
    load_config as load_hybrid_config,
)
from .metrics import mean_metrics, metrics_for_ranking
from .reranker import (
    predict_with_cache,
    ranking_from_candidate_order,
    ranking_from_candidate_scores,
    score_artifact_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reranker.toml"
DEFAULT_SCORE_CACHE = PROJECT_ROOT / "data" / "processed" / "reranker_scores"


@dataclass(frozen=True)
class RerankerConfig:
    key: str
    model_id: str
    revision: str
    license: str
    max_length: int
    batch_size: int
    candidate_depth: int
    k_values: tuple[int, ...]


@dataclass(frozen=True)
class CandidateQuery:
    text: str
    relevant_ids: frozenset[int]
    candidate_ids: tuple[int, ...]
    retrieval_ms: float
    court: str | None = None
    year: str | None = None


def load_config(path: Path) -> RerankerConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    config = RerankerConfig(
        key=str(raw["model"]["key"]),
        model_id=str(raw["model"]["model_id"]),
        revision=str(raw["model"]["revision"]),
        license=str(raw["model"]["license"]),
        max_length=int(raw["model"]["max_length"]),
        batch_size=int(raw["model"]["batch_size"]),
        candidate_depth=int(raw["reranking"]["candidate_depth"]),
        k_values=tuple(int(value) for value in raw["evaluation"]["k_values"]),
    )
    if config.max_length < 1 or config.batch_size < 1 or config.candidate_depth < 1:
        raise ValueError("model length, batch size, and candidate depth must be positive")
    if not config.k_values or any(value < 1 for value in config.k_values):
        raise ValueError("k values must be non-empty and positive")
    if config.candidate_depth < max(config.k_values):
        raise ValueError("candidate depth must cover the largest evaluation k")
    return config


def load_reranker_model(config: RerankerConfig, model_cache: Path) -> tuple[Any, float]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "install the dense dependency extra before running this command"
        ) from error

    started = time.perf_counter()
    model = CrossEncoder(
        config.model_id,
        revision=config.revision,
        device="cpu",
        cache_folder=str(model_cache),
        max_length=config.max_length,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if model.max_seq_length != config.max_length:
        raise ValueError(
            f"reranker maximum length mismatch: configured {config.max_length}, "
            f"got {model.max_seq_length}"
        )
    return model, elapsed_ms


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def _relevant_distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _breakdown_metrics(
    values: dict[str, list[tuple[dict[str, float], dict[str, float]]]],
) -> dict[str, Any]:
    return {
        name: {
            "queries": len(items),
            "hybrid_metrics": mean_metrics(item[0] for item in items),
            "reranked_metrics": mean_metrics(item[1] for item in items),
        }
        for name, items in sorted(values.items())
    }


def evaluate_candidates(
    model: Any,
    corpus: list[str],
    candidates: list[CandidateQuery],
    config: RerankerConfig,
    *,
    score_cache: Path,
    role: str,
) -> dict[str, Any]:
    queries = [candidate.text for candidate in candidates]
    candidate_matrix = np.asarray(
        [candidate.candidate_ids for candidate in candidates],
        dtype=np.int64,
    )
    if candidate_matrix.shape != (len(candidates), config.candidate_depth):
        raise ValueError("generated candidate matrix does not match configured depth")
    score_artifact = predict_with_cache(
        model,
        queries,
        candidate_matrix,
        corpus,
        cache_dir=score_cache,
        model_key=config.key,
        revision=config.revision,
        role=role,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    inference_ms_per_query = score_artifact.inference_elapsed_ms / len(candidates)

    hybrid_observations: list[dict[str, float]] = []
    reranked_observations: list[dict[str, float]] = []
    total_latencies_ms: list[float] = []
    relevant_counts: list[int] = []
    candidate_recalls: list[float] = []
    candidate_hits: list[float] = []
    by_court: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    by_year: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)

    for candidate, scores in zip(candidates, score_artifact.values, strict=True):
        relevant_ids = set(candidate.relevant_ids)
        relevant_counts.append(len(relevant_ids))
        hybrid_ranking = ranking_from_candidate_order(
            candidate.candidate_ids,
            relevant_ids,
            top_k=max(config.k_values),
        )
        rank_started = time.perf_counter()
        reranked_ranking = ranking_from_candidate_scores(
            np.asarray(scores, dtype=np.float32),
            np.asarray(candidate.candidate_ids, dtype=np.int64),
            relevant_ids,
            top_k=max(config.k_values),
        )
        rerank_ms = (time.perf_counter() - rank_started) * 1000
        hybrid_metrics = metrics_for_ranking(hybrid_ranking, relevant_ids, config.k_values)
        reranked_metrics = metrics_for_ranking(reranked_ranking, relevant_ids, config.k_values)
        hybrid_observations.append(hybrid_metrics)
        reranked_observations.append(reranked_metrics)
        retrieved_relevant = sum(
            identifier in relevant_ids for identifier in candidate.candidate_ids
        )
        candidate_recalls.append(retrieved_relevant / len(relevant_ids))
        candidate_hits.append(float(retrieved_relevant > 0))
        total_latencies_ms.append(candidate.retrieval_ms + inference_ms_per_query + rerank_ms)
        if candidate.court is not None:
            by_court[candidate.court].append((hybrid_metrics, reranked_metrics))
        if candidate.year is not None:
            by_year[candidate.year].append((hybrid_metrics, reranked_metrics))

    result: dict[str, Any] = {
        "queries": len(candidates),
        "hybrid_metrics": mean_metrics(hybrid_observations),
        "reranked_metrics": mean_metrics(reranked_observations),
        f"candidate_recall_at_{config.candidate_depth}": statistics.fmean(candidate_recalls),
        f"candidate_hit_rate_at_{config.candidate_depth}": statistics.fmean(candidate_hits),
        "relevant_documents_per_query": _relevant_distribution(relevant_counts),
        "latency_ms": {
            "hybrid_candidate_generation": _distribution(
                [candidate.retrieval_ms for candidate in candidates]
            ),
            "reranker_inference": {
                "total": score_artifact.inference_elapsed_ms,
                "mean_per_query": inference_ms_per_query,
            },
            "total_with_amortized_inference": _distribution(total_latencies_ms),
        },
        "score_cache": score_artifact_metadata(score_artifact),
    }
    if by_court or by_year:
        result["breakdowns"] = {
            "court": _breakdown_metrics(by_court),
            "year": _breakdown_metrics(by_year),
        }
    return result


def generate_pooled_candidates(
    index: BM25Index,
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    pools: dict[str, dict[str, Any]],
    query_field: str,
    query_prefix: str,
    hybrid_config: HybridConfig,
    reranker_config: RerankerConfig,
    max_queries: int | None,
) -> list[CandidateQuery]:
    generated: list[CandidateQuery] = []
    pool_items = sorted(pools.items(), key=lambda item: int(item[0]))
    if max_queries is not None:
        pool_items = pool_items[:max_queries]

    for _, item in pool_items:
        started = time.perf_counter()
        candidate_ids = np.asarray(item["pool"], dtype=np.int64)
        relevant_ids = {int(item["correct_case_id"])}
        query_text = str(item[query_field])
        bm25_ranking = index.rank(
            query_text,
            relevant_ids,
            top_k=min(hybrid_config.component_depth, len(candidate_ids)),
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
            top_k=min(hybrid_config.component_depth, len(candidate_ids)),
        )
        hybrid_ranking = fuse_rankings(
            bm25_ranking,
            dense_ranking,
            relevant_ids,
            hybrid_config,
            top_k=reranker_config.candidate_depth,
        )
        generated.append(
            CandidateQuery(
                text=query_text,
                relevant_ids=frozenset(relevant_ids),
                candidate_ids=hybrid_ranking.top_indices,
                retrieval_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return generated


def run_pooled(
    data_dir: Path,
    dense_model: Any,
    reranker_model: Any,
    dense_config: DenseModelConfig,
    hybrid_config: HybridConfig,
    reranker_config: RerankerConfig,
    *,
    embedding_cache: Path,
    score_cache: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    corpus = build_lookup_corpus(data_dir)
    started = time.perf_counter()
    index = BM25Index(corpus, k1=hybrid_config.k1, b=hybrid_config.b)
    index_build_ms = (time.perf_counter() - started) * 1000
    modes = load_pooled_modes(data_dir)
    prepared_queries = prepare_pooled_query_texts(
        modes,
        dense_config.query_prefix,
        max_queries,
    )
    corpus_artifact = encode_with_cache(
        dense_model,
        corpus,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_corpus",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
    )
    query_artifact = encode_with_cache(
        dense_model,
        prepared_queries,
        cache_dir=embedding_cache,
        model_key=dense_config.key,
        revision=dense_config.revision,
        role="pooled_queries",
        batch_size=dense_config.encode_batch_size,
        dimensions=dense_config.dimensions,
    )
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    results: dict[str, Any] = {}
    for mode, (pools, field_name) in modes.items():
        print(f"generating hybrid candidates for {mode}")
        generated = generate_pooled_candidates(
            index,
            corpus_artifact.values,
            query_artifact.values,
            query_to_index,
            pools,
            field_name,
            dense_config.query_prefix,
            hybrid_config,
            reranker_config,
            max_queries,
        )
        print(f"scoring reranker pairs for {mode}")
        results[mode] = evaluate_candidates(
            reranker_model,
            corpus,
            generated,
            reranker_config,
            score_cache=score_cache,
            role=f"pooled_{mode}",
        )
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
        "modes": results,
    }


def generate_full_candidates(
    index: BM25Index,
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_to_index: dict[str, int],
    candidate_to_index: dict[str, int],
    queries: list[QueryRecord],
    query_prefix: str,
    hybrid_config: HybridConfig,
    reranker_config: RerankerConfig,
) -> list[CandidateQuery]:
    generated: list[CandidateQuery] = []
    candidate_ids = np.arange(len(corpus_embeddings), dtype=np.int64)
    corpus_transposed = corpus_embeddings.T

    for offset in range(0, len(queries), hybrid_config.score_batch_size):
        batch = queries[offset : offset + hybrid_config.score_batch_size]
        query_indices = [query_to_index[f"{query_prefix}{query.text}"] for query in batch]
        score_started = time.perf_counter()
        score_rows = np.asarray(
            query_embeddings[query_indices] @ corpus_transposed, dtype=np.float32
        )
        score_ms_per_query = (time.perf_counter() - score_started) * 1000 / len(batch)

        for query, dense_scores in zip(batch, score_rows, strict=True):
            started = time.perf_counter()
            relevant_ids = {candidate_to_index[text] for text in query.relevant_texts}
            bm25_ranking = index.rank(
                query.text,
                relevant_ids,
                top_k=min(hybrid_config.component_depth, len(candidate_ids)),
            )
            dense_ranking = ranking_from_scores(
                dense_scores,
                candidate_ids,
                relevant_ids,
                top_k=min(hybrid_config.component_depth, len(candidate_ids)),
                candidate_ids_validated=True,
            )
            hybrid_ranking = fuse_rankings(
                bm25_ranking,
                dense_ranking,
                relevant_ids,
                hybrid_config,
                top_k=reranker_config.candidate_depth,
            )
            generated.append(
                CandidateQuery(
                    text=query.text,
                    relevant_ids=frozenset(relevant_ids),
                    candidate_ids=hybrid_ranking.top_indices,
                    retrieval_ms=score_ms_per_query + (time.perf_counter() - started) * 1000,
                    court=query.court,
                    year=query.year,
                )
            )
    return generated


def run_full(
    data_dir: Path,
    split_path: Path,
    dense_model: Any,
    reranker_model: Any,
    dense_config: DenseModelConfig,
    hybrid_config: HybridConfig,
    reranker_config: RerankerConfig,
    *,
    embedding_cache: Path,
    score_cache: Path,
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
        dense_model,
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
        dense_model,
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
    index = BM25Index(corpus, k1=hybrid_config.k1, b=hybrid_config.b)
    index_build_ms = (time.perf_counter() - started) * 1000
    candidate_to_index = {text: index for index, text in enumerate(corpus)}
    query_to_index = query_embedding_lookup(query_artifact, prepared_queries)
    results: dict[str, Any] = {}
    for mode, queries in queries_by_mode.items():
        print(f"generating hybrid candidates for {mode}")
        generated = generate_full_candidates(
            index,
            corpus_artifact.values,
            query_artifact.values,
            query_to_index,
            candidate_to_index,
            queries,
            dense_config.query_prefix,
            hybrid_config,
            reranker_config,
        )
        print(f"scoring reranker pairs for {mode}")
        results[mode] = evaluate_candidates(
            reranker_model,
            corpus,
            generated,
            reranker_config,
            score_cache=score_cache,
            role=f"full_{mode}",
        )
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
        "modes": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid + cross-encoder SG-LegalCite baselines"
    )
    parser.add_argument("--protocol", choices=("pooled", "full"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
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
        reranker_config = load_config(args.config)
        hybrid_config = load_hybrid_config(args.hybrid_config)
        dense_config, _, _ = load_dense_config(
            args.dense_config,
            hybrid_config.dense_model_key,
        )
        dense_model, dense_model_load_ms = load_dense_model(dense_config, args.model_cache)
        reranker_model, reranker_model_load_ms = load_reranker_model(
            reranker_config,
            args.model_cache,
        )
        if args.protocol == "pooled":
            result = run_pooled(
                args.data_dir,
                dense_model,
                reranker_model,
                dense_config,
                hybrid_config,
                reranker_config,
                embedding_cache=args.embedding_cache,
                score_cache=args.score_cache,
                max_queries=args.max_queries,
            )
        else:
            result = run_full(
                args.data_dir,
                args.splits,
                dense_model,
                reranker_model,
                dense_config,
                hybrid_config,
                reranker_config,
                embedding_cache=args.embedding_cache,
                score_cache=args.score_cache,
                max_queries=args.max_queries,
            )
    # This CLI boundary also reports third-party model/download exceptions without a traceback.
    except Exception as error:  # noqa: BLE001
        print(f"reranker benchmark failed: {error}", file=sys.stderr)
        return 1

    result["components"] = {
        "hybrid": asdict(hybrid_config),
        "dense": asdict(dense_config),
        "reranker": asdict(reranker_config),
    }
    result["configuration"] = {
        "device": "cpu",
        "reranker_input": "raw query and raw Cited Case string",
        "reranker_score": "raw single-label cross-encoder logit",
        "tie_break": "ascending candidate ID",
        "max_queries": args.max_queries,
        "dense_model_load_ms": dense_model_load_ms,
        "reranker_model_load_ms": reranker_model_load_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
