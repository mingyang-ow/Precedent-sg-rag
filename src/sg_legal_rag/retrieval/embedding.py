from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class EmbeddingArtifact:
    values: FloatArray
    cache_hit: bool
    elapsed_ms: float
    path: Path
    encoded_rows: int
    reused_rows: int


def texts_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def cache_path(
    cache_dir: Path,
    model_key: str,
    revision: str,
    role: str,
    texts: Sequence[str],
) -> Path:
    digest = texts_digest(texts)[:16]
    return cache_dir / model_key / f"{revision[:12]}_{role}_{digest}.npy"


def encode_with_cache(
    model: Any,
    texts: Sequence[str],
    *,
    cache_dir: Path,
    model_key: str,
    revision: str,
    role: str,
    batch_size: int,
    dimensions: int,
    reuse_texts: Sequence[str] | None = None,
    reuse_values: FloatArray | None = None,
) -> EmbeddingArtifact:
    path = cache_path(cache_dir, model_key, revision, role, texts)
    started = time.perf_counter()
    if path.exists():
        values = np.load(path, mmap_mode="r")
        if values.shape != (len(texts), dimensions):
            raise ValueError(f"cached embedding shape mismatch: {path}")
        return EmbeddingArtifact(
            values=values,
            cache_hit=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            path=path,
            encoded_rows=0,
            reused_rows=0,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if (reuse_texts is None) != (reuse_values is None):
        raise ValueError("reuse texts and values must be supplied together")
    values = np.empty((len(texts), dimensions), dtype=np.float32)
    reused_rows = 0
    missing_positions: list[int] = []
    if reuse_texts is not None and reuse_values is not None:
        if reuse_values.shape != (len(reuse_texts), dimensions):
            raise ValueError("reusable embedding shape mismatch")
        reusable_indices = {text: index for index, text in enumerate(reuse_texts)}
        for position, text in enumerate(texts):
            reusable_index = reusable_indices.get(text)
            if reusable_index is None:
                missing_positions.append(position)
            else:
                values[position] = reuse_values[reusable_index]
                reused_rows += 1
    else:
        missing_positions = list(range(len(texts)))

    if missing_positions:
        encoded = model.encode(
            [texts[position] for position in missing_positions],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        values[missing_positions] = np.asarray(encoded, dtype=np.float32)
    if values.shape != (len(texts), dimensions):
        raise ValueError(
            f"model returned shape {values.shape}; expected {(len(texts), dimensions)}"
        )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.save(temporary, values, allow_pickle=False)
            temporary.flush()
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return EmbeddingArtifact(
        values=np.load(path, mmap_mode="r"),
        cache_hit=False,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        path=path,
        encoded_rows=len(missing_positions),
        reused_rows=reused_rows,
    )


def load_cached_embeddings(
    texts: Sequence[str],
    *,
    cache_dir: Path,
    model_key: str,
    revision: str,
    role: str,
    dimensions: int,
) -> EmbeddingArtifact | None:
    path = cache_path(cache_dir, model_key, revision, role, texts)
    if not path.exists():
        return None
    started = time.perf_counter()
    values = np.load(path, mmap_mode="r")
    if values.shape != (len(texts), dimensions):
        raise ValueError(f"cached embedding shape mismatch: {path}")
    return EmbeddingArtifact(
        values=values,
        cache_hit=True,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        path=path,
        encoded_rows=0,
        reused_rows=0,
    )


def artifact_metadata(artifact: EmbeddingArtifact) -> dict[str, Any]:
    return {
        "cache_hit": artifact.cache_hit,
        "elapsed_ms": artifact.elapsed_ms,
        "shape": list(artifact.values.shape),
        "cache_file": artifact.path.name,
        "encoded_rows": artifact.encoded_rows,
        "reused_rows": artifact.reused_rows,
    }
