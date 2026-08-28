from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bm25 import Ranking
from .embedding import texts_digest

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ScoreArtifact:
    values: FloatArray
    cache_hit: bool
    inference_elapsed_ms: float
    path: Path
    pairs: int


def ranking_from_candidate_scores(
    scores: FloatArray,
    candidate_ids: IntArray,
    relevant_ids: set[int],
    *,
    top_k: int,
) -> Ranking:
    if scores.ndim != 1 or candidate_ids.ndim != 1 or len(scores) != len(candidate_ids):
        raise ValueError("scores and candidate IDs must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("reranker scores must be finite")
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    if not relevant_ids:
        raise ValueError("at least one relevant candidate is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    order = np.lexsort((candidate_ids, -scores))
    ranked = tuple(int(identifier) for identifier in candidate_ids[order])
    first_relevant_rank = next(
        (rank for rank, identifier in enumerate(ranked, start=1) if identifier in relevant_ids),
        None,
    )
    return Ranking(
        top_indices=ranked[:top_k],
        first_relevant_rank=first_relevant_rank,
        positive_matches=len(ranked),
    )


def ranking_from_candidate_order(
    candidate_ids: Sequence[int], relevant_ids: set[int], *, top_k: int
) -> Ranking:
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    if not relevant_ids:
        raise ValueError("at least one relevant candidate is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    first_relevant_rank = next(
        (
            rank
            for rank, identifier in enumerate(candidate_ids, start=1)
            if identifier in relevant_ids
        ),
        None,
    )
    return Ranking(
        top_indices=tuple(candidate_ids[:top_k]),
        first_relevant_rank=first_relevant_rank,
        positive_matches=len(candidate_ids),
    )


def _update_text_digest(digest: Any, text: str) -> None:
    encoded = text.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def score_cache_path(
    cache_dir: Path,
    *,
    model_key: str,
    revision: str,
    role: str,
    max_length: int,
    queries: Sequence[str],
    candidate_ids: IntArray,
    corpus: Sequence[str],
) -> Path:
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != len(queries):
        raise ValueError("candidate matrix rows must align with queries")
    digest = hashlib.sha256()
    _update_text_digest(digest, str(max_length))
    _update_text_digest(digest, texts_digest(corpus))
    for query in queries:
        _update_text_digest(digest, query)
    contiguous_ids = np.ascontiguousarray(candidate_ids, dtype="<i8")
    digest.update(contiguous_ids.tobytes())
    return cache_dir / model_key / f"{revision[:12]}_{role}_{digest.hexdigest()[:16]}.npy"


def _load_metadata(path: Path, expected_pairs: int) -> float:
    metadata_path = path.with_suffix(".json")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if int(metadata["pairs"]) != expected_pairs:
        raise ValueError(f"cached score pair count mismatch: {metadata_path}")
    elapsed_ms = float(metadata["inference_elapsed_ms"])
    if not np.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError(f"cached inference time is invalid: {metadata_path}")
    return elapsed_ms


def predict_with_cache(
    model: Any,
    queries: Sequence[str],
    candidate_ids: IntArray,
    corpus: Sequence[str],
    *,
    cache_dir: Path,
    model_key: str,
    revision: str,
    role: str,
    batch_size: int,
    max_length: int,
) -> ScoreArtifact:
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != len(queries):
        raise ValueError("candidate matrix rows must align with queries")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(corpus)):
        raise IndexError("candidate ID out of corpus range")

    path = score_cache_path(
        cache_dir,
        model_key=model_key,
        revision=revision,
        role=role,
        max_length=max_length,
        queries=queries,
        candidate_ids=candidate_ids,
        corpus=corpus,
    )
    expected_shape = candidate_ids.shape
    pairs_count = int(candidate_ids.size)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        values = np.load(path, mmap_mode="r")
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"cached reranker score shape or values are invalid: {path}")
        return ScoreArtifact(
            values=values,
            cache_hit=True,
            inference_elapsed_ms=_load_metadata(path, pairs_count),
            path=path,
            pairs=pairs_count,
        )

    started = time.perf_counter()
    pairs = [
        (query, corpus[int(candidate_id)])
        for query, row in zip(queries, candidate_ids, strict=True)
        for candidate_id in row
    ]
    predicted = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    inference_elapsed_ms = (time.perf_counter() - started) * 1000
    values = np.asarray(predicted, dtype=np.float32).reshape(expected_shape)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError("reranker returned an invalid score matrix")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.save(temporary, values, allow_pickle=False)
            temporary.flush()
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    metadata = {
        "pairs": pairs_count,
        "inference_elapsed_ms": inference_elapsed_ms,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        metadata_temporary_path = Path(temporary.name)
        try:
            json.dump(metadata, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            metadata_temporary_path.replace(metadata_path)
        except BaseException:
            metadata_temporary_path.unlink(missing_ok=True)
            raise
    return ScoreArtifact(
        values=np.load(path, mmap_mode="r"),
        cache_hit=False,
        inference_elapsed_ms=inference_elapsed_ms,
        path=path,
        pairs=pairs_count,
    )


def score_artifact_metadata(artifact: ScoreArtifact) -> dict[str, Any]:
    return {
        "cache_hit": artifact.cache_hit,
        "cache_file": artifact.path.name,
        "shape": list(artifact.values.shape),
        "pairs": artifact.pairs,
        "inference_elapsed_ms": artifact.inference_elapsed_ms,
        "pairs_per_second": artifact.pairs / (artifact.inference_elapsed_ms / 1000),
    }
