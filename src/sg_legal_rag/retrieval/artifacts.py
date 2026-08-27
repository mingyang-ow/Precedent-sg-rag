from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from sg_legal_rag.ingestion.manifest import load_manifest

from .bm25 import BM25Index
from .corpus_repair import CorpusRepairDataset, HistoricalContext, load_corpus_repair_dataset
from .corpus_repair_benchmark import DEFAULT_CONFIG, load_config
from .tokenization import TOKENIZATION_VERSION, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "splits_temporal.csv"
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "configs" / "dataset_manifest.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "retrieval-artifacts"

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_VERSION = "passage-bm25-v1"
RETRIEVAL_IMPLEMENTATION_VERSION = "canonical-passage-max-bm25-v1"
PRODUCTION_BM25_K1 = 1.2
PRODUCTION_BM25_B = 0.75
CORPUS_FORMAT = "precedent-runtime-corpus-jsonl-v1"
BM25_FORMAT = "precedent-bm25-numpy-v1"
MANIFEST_FILE = "manifest.json"
CORPUS_FILE = "corpus.jsonl.gz"
BM25_TERMS_FILE = "bm25-terms.txt.gz"
BM25_DOCUMENT_LENGTHS_FILE = "bm25-document-lengths.npy"
BM25_IDF_FILE = "bm25-idf.npy"
BM25_POSTING_OFFSETS_FILE = "bm25-posting-offsets.npy"
BM25_POSTING_DOCUMENT_IDS_FILE = "bm25-posting-document-ids.npy"
BM25_POSTING_FREQUENCIES_FILE = "bm25-posting-frequencies.npy"
BM25_FILES = {
    "terms": BM25_TERMS_FILE,
    "document_lengths": BM25_DOCUMENT_LENGTHS_FILE,
    "idf": BM25_IDF_FILE,
    "posting_offsets": BM25_POSTING_OFFSETS_FILE,
    "posting_document_ids": BM25_POSTING_DOCUMENT_IDS_FILE,
    "posting_frequencies": BM25_POSTING_FREQUENCIES_FILE,
}


class RetrievalArtifactError(RuntimeError):
    """Raised when a prepared retrieval bundle cannot be trusted or loaded."""


@dataclass(frozen=True)
class RuntimePassageCorpus:
    case_texts: tuple[str, ...]
    contexts: tuple[HistoricalContext, ...]
    context_case_ids: np.ndarray


class RuntimeBM25Index:
    """Compact read-only full-corpus BM25 state backed by NumPy arrays."""

    def __init__(
        self,
        *,
        terms: tuple[str, ...],
        document_lengths: np.ndarray,
        idf: np.ndarray,
        posting_offsets: np.ndarray,
        posting_document_ids: np.ndarray,
        posting_frequencies: np.ndarray,
        k1: float,
        b: float,
        average_document_length: float,
    ) -> None:
        self.terms = terms
        self.term_indices = {term: index for index, term in enumerate(terms)}
        self.document_lengths = document_lengths
        self.idf = idf
        self.posting_offsets = posting_offsets
        self.posting_document_ids = posting_document_ids
        self.posting_frequencies = posting_frequencies
        self.k1 = k1
        self.b = b
        self.average_document_length = average_document_length
        self.document_count = len(document_lengths)

    def scores(self, query: str) -> dict[int, float]:
        query_terms = tuple(sorted(set(tokenize(query)).intersection(self.term_indices)))
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            term_index = self.term_indices[term]
            start = int(self.posting_offsets[term_index])
            stop = int(self.posting_offsets[term_index + 1])
            term_idf = float(self.idf[term_index])
            for offset in range(start, stop):
                document_index = int(self.posting_document_ids[offset])
                frequency = int(self.posting_frequencies[offset])
                length = int(self.document_lengths[document_index])
                normalization = self.k1 * (
                    1.0 - self.b + self.b * length / self.average_document_length
                )
                scores[document_index] += (
                    term_idf * (frequency * (self.k1 + 1.0)) / (frequency + normalization)
                )
        return dict(scores)


@dataclass(frozen=True)
class RetrievalArtifactIdentity:
    artifact_version: str
    manifest_digest: str
    document_count: int
    load_ms: float


@dataclass(frozen=True)
class LoadedRetrievalArtifacts:
    corpus: RuntimePassageCorpus
    index: RuntimeBM25Index
    identity: RetrievalArtifactIdentity


@dataclass(frozen=True)
class BuiltRetrievalArtifacts:
    output_dir: Path
    manifest_digest: str
    document_count: int
    case_count: int
    build_ms: float
    source_verification_ms: float | None = None
    corpus_build_ms: float | None = None
    index_build_ms: float | None = None
    artifact_write_validate_ms: float | None = None
    equivalence_queries: int = 0


@dataclass(frozen=True)
class RetrievalBuildProvenance:
    dataset_id: str
    dataset_revision: str
    source_file: str
    source_digest: str
    source_size: int
    config_digest: str
    splits_digest: str
    source_date_epoch: int = 0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class _DeterministicGzipJsonLines:
    def __init__(self, path: Path) -> None:
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._payload_digest = hashlib.sha256()
        self.payload_bytes = 0
        self.records = 0

    def write(self, value: object) -> None:
        encoded = _canonical_json(value) + b"\n"
        self._gzip.write(encoded)
        self._payload_digest.update(encoded)
        self.payload_bytes += len(encoded)
        self.records += 1

    def close(self) -> str:
        self._gzip.close()
        self._raw.close()
        return self._payload_digest.hexdigest()


def _iter_json_lines(path: Path, digest: Any) -> Iterator[dict[str, Any]]:
    try:
        with gzip.open(path, "rb") as stream:
            for number, line in enumerate(stream, start=1):
                digest.update(line)
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RetrievalArtifactError(f"{path.name} record {number} is not an object")
                yield value
    except RetrievalArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalArtifactError(f"{path.name} is unreadable or malformed") from error


def _created_at(source_date_epoch: int) -> str:
    if source_date_epoch < 0:
        raise ValueError("source date epoch cannot be negative")
    return datetime.fromtimestamp(source_date_epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _write_corpus(path: Path, dataset: CorpusRepairDataset) -> dict[str, object]:
    writer = _DeterministicGzipJsonLines(path)
    try:
        writer.write(
            {
                "case_texts": dataset.case_texts,
                "format": CORPUS_FORMAT,
                "max_passage_chars": dataset.max_passage_chars,
                "record_type": "metadata",
            }
        )
        for context, case_id in zip(dataset.contexts, dataset.context_case_ids, strict=True):
            payload = asdict(context)
            # Runtime evidence integrity binds the exact bounded passage, not the source paragraph.
            payload["digest"] = hashlib.sha256(context.text.encode("utf-8")).hexdigest()
            writer.write(
                {
                    "case_id": int(case_id),
                    "context": payload,
                    "record_type": "passage",
                }
            )
        payload_digest = writer.close()
    except BaseException:
        if not writer._gzip.closed:
            writer._gzip.close()
            writer._raw.close()
        raise
    return {
        "file": path.name,
        "file_sha256": _sha256_file(path),
        "file_size": path.stat().st_size,
        "format": CORPUS_FORMAT,
        "payload_sha256": payload_digest,
        "payload_size": writer.payload_bytes,
        "record_count": writer.records,
    }


def _file_metadata(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "file_sha256": _sha256_file(path),
        "file_size": path.stat().st_size,
    }


def _write_bm25(directory: Path, index: BM25Index) -> dict[str, object]:
    terms = tuple(sorted(index.postings))
    posting_count = sum(len(index.postings[term]) for term in terms)
    document_lengths = np.asarray(index.document_lengths, dtype="<i4")
    idf = np.asarray([index.idf[term] for term in terms], dtype="<f8")
    posting_offsets = np.empty(len(terms) + 1, dtype="<i8")
    posting_document_ids = np.empty(posting_count, dtype="<i4")
    posting_frequencies = np.empty(posting_count, dtype="<u4")
    cursor = 0
    for term_index, term in enumerate(terms):
        posting_offsets[term_index] = cursor
        for document_index, frequency in index.postings[term]:
            posting_document_ids[cursor] = document_index
            posting_frequencies[cursor] = frequency
            cursor += 1
    posting_offsets[-1] = cursor

    terms_path = directory / BM25_TERMS_FILE
    raw = terms_path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    try:
        for term in terms:
            compressed.write(term.encode("utf-8") + b"\n")
    finally:
        compressed.close()
        raw.close()

    arrays = {
        "document_lengths": document_lengths,
        "idf": idf,
        "posting_offsets": posting_offsets,
        "posting_document_ids": posting_document_ids,
        "posting_frequencies": posting_frequencies,
    }
    for role, values in arrays.items():
        with (directory / BM25_FILES[role]).open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
    return {
        "average_document_length": index.average_document_length,
        "b": index.b,
        "document_count": index.document_count,
        "files": {
            role: _file_metadata(directory / file_name) for role, file_name in BM25_FILES.items()
        },
        "format": BM25_FORMAT,
        "k1": index.k1,
        "posting_count": posting_count,
        "term_count": len(terms),
    }


def _expect_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RetrievalArtifactError(f"artifact manifest {name} is invalid")
    return value


def _read_manifest(bundle_dir: Path) -> tuple[dict[str, Any], str]:
    path = bundle_dir / MANIFEST_FILE
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RetrievalArtifactError(
            "retrieval artifact manifest is missing or unreadable"
        ) from error
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalArtifactError("retrieval artifact manifest is malformed") from error
    if not isinstance(manifest, dict) or raw != _canonical_json(manifest):
        raise RetrievalArtifactError("retrieval artifact manifest is not canonical")
    return manifest, hashlib.sha256(raw).hexdigest()


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    expected = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "retrieval_implementation_version": RETRIEVAL_IMPLEMENTATION_VERSION,
        "tokenization_version": TOKENIZATION_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RetrievalArtifactError(f"retrieval artifact {field} is incompatible")
    if not isinstance(manifest.get("created_at"), str):
        raise RetrievalArtifactError("retrieval artifact created_at is invalid")
    document_count = manifest.get("document_count")
    if (
        not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or document_count < 1
    ):
        raise RetrievalArtifactError("retrieval artifact document_count is invalid")
    bm25 = _expect_mapping(manifest.get("bm25"), "bm25")
    if bm25.get("format") != BM25_FORMAT:
        raise RetrievalArtifactError("retrieval artifact BM25 format is incompatible")
    k1, b = bm25.get("k1"), bm25.get("b")
    if not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
        raise RetrievalArtifactError("retrieval artifact BM25 k1 is invalid")
    if not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
        raise RetrievalArtifactError("retrieval artifact BM25 b is invalid")
    if k1 != PRODUCTION_BM25_K1 or b != PRODUCTION_BM25_B:
        raise RetrievalArtifactError("retrieval artifact BM25 parameters are incompatible")
    average_length = bm25.get("average_document_length")
    if not isinstance(average_length, (int, float)) or not math.isfinite(average_length):
        raise RetrievalArtifactError("retrieval artifact BM25 average length is invalid")
    for field in ("term_count", "posting_count"):
        value = bm25.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RetrievalArtifactError(f"retrieval artifact BM25 {field} is invalid")
    corpus = _expect_mapping(manifest.get("corpus"), "corpus")
    if corpus.get("format") != CORPUS_FORMAT:
        raise RetrievalArtifactError("retrieval artifact corpus format is incompatible")
    if corpus.get("file") != CORPUS_FILE:
        raise RetrievalArtifactError("retrieval artifact file name is incompatible")
    for field in ("file_sha256", "payload_sha256"):
        value = corpus.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RetrievalArtifactError(f"retrieval artifact {field} is invalid")
    files = _expect_mapping(bm25.get("files"), "bm25 files")
    if set(files) != set(BM25_FILES):
        raise RetrievalArtifactError("retrieval artifact BM25 file roles are incompatible")
    for role, expected_file in BM25_FILES.items():
        metadata = _expect_mapping(files[role], f"bm25 file {role}")
        if metadata.get("file") != expected_file:
            raise RetrievalArtifactError("retrieval artifact BM25 file name is incompatible")
        digest = metadata.get("file_sha256")
        size = metadata.get("file_size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RetrievalArtifactError("retrieval artifact BM25 file digest is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise RetrievalArtifactError("retrieval artifact BM25 file size is invalid")


def _verify_file(bundle_dir: Path, section: Mapping[str, Any]) -> Path:
    file_name = str(section["file"])
    path = bundle_dir / file_name
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RetrievalArtifactError(f"retrieval artifact {file_name} is missing") from error
    if size != section.get("file_size") or _sha256_file(path) != section["file_sha256"]:
        raise RetrievalArtifactError(f"retrieval artifact {file_name} failed file integrity")
    return path


def _load_corpus(path: Path, section: Mapping[str, Any]) -> RuntimePassageCorpus:
    digest = hashlib.sha256()
    records = _iter_json_lines(path, digest)
    try:
        metadata = next(records)
    except StopIteration as error:
        raise RetrievalArtifactError("retrieval corpus is empty") from error
    if metadata.get("record_type") != "metadata" or metadata.get("format") != CORPUS_FORMAT:
        raise RetrievalArtifactError("retrieval corpus metadata is incompatible")
    case_texts_raw = metadata.get("case_texts")
    if not isinstance(case_texts_raw, list) or not case_texts_raw:
        raise RetrievalArtifactError("retrieval corpus case table is invalid")
    case_texts = tuple(str(value) for value in case_texts_raw)
    contexts: list[HistoricalContext] = []
    case_ids: list[int] = []
    for record in records:
        if record.get("record_type") != "passage":
            raise RetrievalArtifactError("retrieval corpus contains an unexpected record")
        case_id = record.get("case_id")
        if (
            not isinstance(case_id, int)
            or isinstance(case_id, bool)
            or not 0 <= case_id < len(case_texts)
        ):
            raise RetrievalArtifactError("retrieval corpus case ID is out of range")
        context = _expect_mapping(record.get("context"), "corpus context")
        try:
            loaded_context = HistoricalContext(**context)
        except (TypeError, ValueError) as error:
            raise RetrievalArtifactError("retrieval corpus context is invalid") from error
        expected_digest = hashlib.sha256(loaded_context.text.encode("utf-8")).hexdigest()
        if loaded_context.digest != expected_digest:
            raise RetrievalArtifactError("retrieval corpus passage digest is inconsistent")
        contexts.append(loaded_context)
        case_ids.append(case_id)
    if digest.hexdigest() != section["payload_sha256"]:
        raise RetrievalArtifactError("retrieval corpus failed payload integrity")
    if len(contexts) != section.get("document_count"):
        raise RetrievalArtifactError("retrieval corpus document count is inconsistent")
    context_case_ids = np.asarray(case_ids, dtype=np.int64)
    context_case_ids.setflags(write=False)
    return RuntimePassageCorpus(
        case_texts=case_texts,
        contexts=tuple(contexts),
        context_case_ids=context_case_ids,
    )


def _load_bm25(bundle_dir: Path, section: Mapping[str, Any]) -> RuntimeBM25Index:
    files = _expect_mapping(section["files"], "bm25 files")
    paths = {
        role: _verify_file(bundle_dir, _expect_mapping(files[role], f"bm25 file {role}"))
        for role in BM25_FILES
    }
    try:
        with gzip.open(paths["terms"], "rt", encoding="utf-8", newline="") as stream:
            terms = tuple(line.removesuffix("\n") for line in stream)
        arrays = {
            role: np.load(paths[role], allow_pickle=False, mmap_mode="r")
            for role in BM25_FILES
            if role != "terms"
        }
    except (OSError, UnicodeError, ValueError) as error:
        raise RetrievalArtifactError("retrieval BM25 state is unreadable") from error
    document_lengths = arrays["document_lengths"]
    idf = arrays["idf"]
    posting_offsets = arrays["posting_offsets"]
    posting_document_ids = arrays["posting_document_ids"]
    posting_frequencies = arrays["posting_frequencies"]
    document_count = int(section["document_count"])
    term_count = int(section["term_count"])
    posting_count = int(section["posting_count"])
    if (
        document_lengths.dtype != np.dtype("<i4")
        or idf.dtype != np.dtype("<f8")
        or posting_offsets.dtype != np.dtype("<i8")
        or posting_document_ids.dtype != np.dtype("<i4")
        or posting_frequencies.dtype != np.dtype("<u4")
    ):
        raise RetrievalArtifactError("retrieval BM25 array type is incompatible")
    if any(values.ndim != 1 for values in arrays.values()):
        raise RetrievalArtifactError("retrieval BM25 arrays must be one-dimensional")
    if (
        len(document_lengths) != document_count
        or len(terms) != term_count
        or len(idf) != term_count
        or len(posting_offsets) != term_count + 1
        or len(posting_document_ids) != posting_count
        or len(posting_frequencies) != posting_count
    ):
        raise RetrievalArtifactError("retrieval BM25 array lengths are inconsistent")
    if (
        not terms
        or any(not term for term in terms)
        or any(left >= right for left, right in pairwise(terms))
    ):
        raise RetrievalArtifactError("retrieval BM25 term table is not canonical")
    if (
        int(posting_offsets[0]) != 0
        or int(posting_offsets[-1]) != posting_count
        or np.any(np.diff(posting_offsets) <= 0)
    ):
        raise RetrievalArtifactError("retrieval BM25 posting offsets are inconsistent")
    if (
        np.any(document_lengths < 0)
        or not np.isfinite(idf).all()
        or np.any(posting_document_ids < 0)
        or np.any(posting_document_ids >= document_count)
        or np.any(posting_frequencies < 1)
    ):
        raise RetrievalArtifactError("retrieval BM25 values are out of range")
    observed_lengths = np.bincount(
        posting_document_ids,
        weights=posting_frequencies,
        minlength=document_count,
    )
    if not np.array_equal(observed_lengths, document_lengths):
        raise RetrievalArtifactError("retrieval BM25 document lengths are inconsistent")
    average_document_length = float(section["average_document_length"])
    if average_document_length != float(document_lengths.sum()) / document_count:
        raise RetrievalArtifactError("retrieval BM25 average document length is inconsistent")
    return RuntimeBM25Index(
        terms=terms,
        document_lengths=document_lengths,
        idf=idf,
        posting_offsets=posting_offsets,
        posting_document_ids=posting_document_ids,
        posting_frequencies=posting_frequencies,
        k1=float(section["k1"]),
        b=float(section["b"]),
        average_document_length=average_document_length,
    )


def load_retrieval_artifacts(bundle_dir: Path) -> LoadedRetrievalArtifacts:
    started = time.perf_counter()
    manifest, manifest_digest = _read_manifest(bundle_dir)
    _validate_manifest(manifest)
    corpus_section = _expect_mapping(manifest["corpus"], "corpus")
    bm25_section = _expect_mapping(manifest["bm25"], "bm25")
    corpus_path = _verify_file(bundle_dir, corpus_section)
    corpus = _load_corpus(corpus_path, corpus_section)
    index = _load_bm25(bundle_dir, bm25_section)
    document_count = int(manifest["document_count"])
    if len(corpus.contexts) != document_count or index.document_count != document_count:
        raise RetrievalArtifactError("retrieval artifact components are not aligned")
    if index.k1 != bm25_section["k1"] or index.b != bm25_section["b"]:
        raise RetrievalArtifactError("retrieval artifact BM25 parameters are inconsistent")
    return LoadedRetrievalArtifacts(
        corpus=corpus,
        index=index,
        identity=RetrievalArtifactIdentity(
            artifact_version=ARTIFACT_VERSION,
            manifest_digest=manifest_digest,
            document_count=document_count,
            load_ms=(time.perf_counter() - started) * 1000,
        ),
    )


def _publish_staged_bundle(staging: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        staging.replace(output_dir)
        return
    backup = output_dir.with_name(f".{output_dir.name}.previous-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"artifact backup already exists: {backup.name}")
    output_dir.replace(backup)
    try:
        staging.replace(output_dir)
    except BaseException:
        backup.replace(output_dir)
        raise
    shutil.rmtree(backup)


def _validate_score_equivalence(canonical: BM25Index, loaded: RuntimeBM25Index) -> int:
    terms = sorted(canonical.postings, key=lambda term: (len(canonical.postings[term]), term))
    positions = sorted({0, len(terms) // 4, len(terms) // 2, 3 * len(terms) // 4, len(terms) - 1})
    selected = [terms[position] for position in positions]
    queries = (*selected, " ".join(selected))
    for query in queries:
        if loaded.scores(query) != canonical.scores(query):
            raise RetrievalArtifactError("loaded BM25 scores differ from canonical BM25")
    return len(queries)


def write_retrieval_bundle(
    *,
    dataset: CorpusRepairDataset,
    index: BM25Index,
    provenance: RetrievalBuildProvenance,
    output_dir: Path,
) -> BuiltRetrievalArtifacts:
    """Serialize, self-validate, and publish a prepared retrieval bundle."""

    started = time.perf_counter()
    if len(dataset.contexts) != index.document_count:
        raise ValueError("retrieval corpus and BM25 document counts are not aligned")
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        corpus = _write_corpus(staging / CORPUS_FILE, dataset)
        corpus["document_count"] = len(dataset.contexts)
        corpus["case_count"] = len(dataset.case_texts)
        bm25 = _write_bm25(staging, index)
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "bm25": bm25,
            "corpus": corpus,
            "created_at": _created_at(provenance.source_date_epoch),
            "dataset": {
                "dataset_id": provenance.dataset_id,
                "revision": provenance.dataset_revision,
            },
            "document_count": len(dataset.contexts),
            "inputs": {
                "config_sha256": provenance.config_digest,
                "splits_sha256": provenance.splits_digest,
            },
            "retrieval_implementation_version": RETRIEVAL_IMPLEMENTATION_VERSION,
            "source": {
                "file": provenance.source_file,
                "sha256": provenance.source_digest,
                "size": provenance.source_size,
            },
            "tokenization_version": TOKENIZATION_VERSION,
        }
        manifest_bytes = _canonical_json(manifest)
        (staging / MANIFEST_FILE).write_bytes(manifest_bytes)
        loaded = load_retrieval_artifacts(staging)
        equivalence_queries = _validate_score_equivalence(index, loaded.index)
        _publish_staged_bundle(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return BuiltRetrievalArtifacts(
        output_dir=output_dir,
        manifest_digest=loaded.identity.manifest_digest,
        document_count=len(dataset.contexts),
        case_count=len(dataset.case_texts),
        build_ms=(time.perf_counter() - started) * 1000,
        equivalence_queries=equivalence_queries,
    )


def build_retrieval_artifacts(
    *,
    data_dir: Path,
    splits_path: Path,
    config_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    source_date_epoch: int = 0,
) -> BuiltRetrievalArtifacts:
    started = time.perf_counter()
    source_path = data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv"
    dataset_manifest = load_manifest(dataset_manifest_path)
    source_entry = next(
        (item for item in dataset_manifest.files if item.name == source_path.name), None
    )
    if source_entry is None or source_entry.sha256 is None:
        raise ValueError("dataset manifest lacks a reproducible core source digest")
    source_digest = _sha256_file(source_path)
    if source_path.stat().st_size != source_entry.size or source_digest != source_entry.sha256:
        raise ValueError("core source does not match the immutable dataset manifest")
    source_verified = time.perf_counter()
    config = load_config(config_path)
    if config.k1 != PRODUCTION_BM25_K1 or config.b != PRODUCTION_BM25_B:
        raise ValueError("corpus config BM25 parameters require a new artifact version")
    dataset = load_corpus_repair_dataset(
        source_path,
        splits_path,
        evidence_cutoff_year=config.evidence_cutoff_year,
        max_passage_chars=config.max_passage_chars,
        max_profile_passages=config.max_profile_passages,
        max_profile_identifier_chars=config.max_profile_identifier_chars,
        max_profile_context_chars=config.max_profile_context_chars,
        max_profile_chars=config.max_profile_chars,
    )
    corpus_built = time.perf_counter()
    index = BM25Index(
        tuple(context.text for context in dataset.contexts),
        k1=config.k1,
        b=config.b,
    )
    index_built = time.perf_counter()
    written = write_retrieval_bundle(
        dataset=dataset,
        index=index,
        provenance=RetrievalBuildProvenance(
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.revision,
            source_file=source_path.name,
            source_digest=source_digest,
            source_size=source_entry.size,
            config_digest=_sha256_file(config_path),
            splits_digest=_sha256_file(splits_path),
            source_date_epoch=source_date_epoch,
        ),
        output_dir=output_dir,
    )
    return BuiltRetrievalArtifacts(
        output_dir=written.output_dir,
        manifest_digest=written.manifest_digest,
        document_count=written.document_count,
        case_count=written.case_count,
        build_ms=(time.perf_counter() - started) * 1000,
        source_verification_ms=(source_verified - started) * 1000,
        corpus_build_ms=(corpus_built - source_verified) * 1000,
        index_build_ms=(index_built - corpus_built) * 1000,
        artifact_write_validate_ms=written.build_ms,
        equivalence_queries=written.equivalence_queries,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate immutable production passage-BM25 artifacts."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "0")),
        help="reproducible manifest timestamp (defaults to the Unix epoch)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        built = build_retrieval_artifacts(
            data_dir=args.data_dir,
            splits_path=args.splits,
            config_path=args.config,
            dataset_manifest_path=args.dataset_manifest,
            output_dir=args.output,
            source_date_epoch=args.source_date_epoch,
        )
    except Exception as error:  # noqa: BLE001
        print(f"retrieval artifact build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_version": ARTIFACT_VERSION,
                "build_ms": round(built.build_ms, 3),
                "case_count": built.case_count,
                "document_count": built.document_count,
                "equivalence_queries": built.equivalence_queries,
                "manifest_digest": built.manifest_digest,
                "output": str(built.output_dir),
                "status": "built_and_validated",
                "timings_ms": {
                    "artifact_write_validate": round(built.artifact_write_validate_ms or 0.0, 3),
                    "corpus_build": round(built.corpus_build_ms or 0.0, 3),
                    "index_build": round(built.index_build_ms or 0.0, 3),
                    "source_verification": round(built.source_verification_ms or 0.0, 3),
                },
            },
            sort_keys=True,
        )
    )
    return 0
