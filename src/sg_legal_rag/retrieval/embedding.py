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

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class EmbeddingArtifact:
    values: FloatArray
    cache_hit: bool
    elapsed_ms: float
    path: Path
    encoded_rows: int
    reused_rows: int
    resumed_rows: int


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


def _partial_cache_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f".{path.name}.partial.npy"),
        path.with_name(f".{path.name}.partial.json"),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(payload, temporary, separators=(",", ":"), sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def _positions_digest(positions: list[int]) -> str:
    values = np.asarray(positions, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


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
            resumed_rows=0,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if (reuse_texts is None) != (reuse_values is None):
        raise ValueError("reuse texts and values must be supplied together")
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
                reused_rows += 1
    else:
        missing_positions = list(range(len(texts)))

    partial_path, progress_path = _partial_cache_paths(path)
    progress_signature = {
        "version": 1,
        "cache_file": path.name,
        "rows": len(texts),
        "dimensions": dimensions,
        "reused_rows": reused_rows,
        "missing_positions_digest": _positions_digest(missing_positions),
    }
    resumed_rows = 0
    if partial_path.exists() and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as stream:
            progress = json.load(stream)
        actual_signature = {key: progress.get(key) for key in progress_signature}
        if actual_signature != progress_signature:
            raise ValueError(f"partial embedding cache signature mismatch: {progress_path}")
        next_missing_offset = progress.get("next_missing_offset")
        if not isinstance(next_missing_offset, int) or not 0 <= next_missing_offset <= len(
            missing_positions
        ):
            raise ValueError(f"invalid partial embedding progress: {progress_path}")
        values = np.lib.format.open_memmap(partial_path, mode="r+")
        if values.shape != (len(texts), dimensions) or values.dtype != np.float32:
            raise ValueError(f"partial embedding shape mismatch: {partial_path}")
        resumed_rows = next_missing_offset
        if resumed_rows:
            print(
                f"resuming embeddings: {resumed_rows}/{len(missing_positions)} encoded rows",
                flush=True,
            )
    else:
        values = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), dimensions),
        )
        if reuse_texts is not None and reuse_values is not None:
            reusable_indices = {text: index for index, text in enumerate(reuse_texts)}
            for position, text in enumerate(texts):
                reusable_index = reusable_indices.get(text)
                if reusable_index is not None:
                    values[position] = reuse_values[reusable_index]
        values.flush()
        next_missing_offset = 0
        _write_json_atomic(
            progress_path,
            {**progress_signature, "next_missing_offset": next_missing_offset},
        )

    checkpoint_rows = max(batch_size, batch_size * 4)
    while next_missing_offset < len(missing_positions):
        chunk_positions = missing_positions[
            next_missing_offset : next_missing_offset + checkpoint_rows
        ]
        encoded = np.asarray(
            model.encode(
                [texts[position] for position in chunk_positions],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        expected_shape = (len(chunk_positions), dimensions)
        if encoded.shape != expected_shape:
            raise ValueError(f"model returned shape {encoded.shape}; expected {expected_shape}")
        values[chunk_positions] = encoded
        values.flush()
        next_missing_offset += len(chunk_positions)
        _write_json_atomic(
            progress_path,
            {**progress_signature, "next_missing_offset": next_missing_offset},
        )
        print(
            f"checkpointed embeddings: {next_missing_offset}/{len(missing_positions)} encoded rows",
            flush=True,
        )

    if values.shape != (len(texts), dimensions):
        raise ValueError(
            f"model returned shape {values.shape}; expected {(len(texts), dimensions)}"
        )
    del values
    partial_path.replace(path)
    progress_path.unlink(missing_ok=True)
    return EmbeddingArtifact(
        values=np.load(path, mmap_mode="r"),
        cache_hit=False,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        path=path,
        encoded_rows=len(missing_positions),
        reused_rows=reused_rows,
        resumed_rows=resumed_rows,
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
        resumed_rows=0,
    )


def artifact_metadata(artifact: EmbeddingArtifact) -> dict[str, Any]:
    return {
        "cache_hit": artifact.cache_hit,
        "elapsed_ms": artifact.elapsed_ms,
        "shape": list(artifact.values.shape),
        "cache_file": artifact.path.name,
        "encoded_rows": artifact.encoded_rows,
        "reused_rows": artifact.reused_rows,
        "resumed_rows": artifact.resumed_rows,
    }
