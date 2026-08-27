from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Protocol

from sg_legal_rag.generation.evidence import EvidenceItem, retrieve_passages
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset, load_corpus_repair_dataset
from sg_legal_rag.retrieval.corpus_repair_benchmark import load_config


class EvidenceRetriever(Protocol):
    def is_ready(self) -> bool: ...

    def retrieve(self, query_text: str, *, top_k: int) -> tuple[EvidenceItem, ...]: ...


class RetrievalUnavailable(RuntimeError):
    pass


class LazyPassageBM25Retriever:
    """Load the leakage-safe historical passage corpus only on the first retrieval request."""

    def __init__(self, *, data_dir: Path, splits_path: Path, config_path: Path) -> None:
        self.data_dir = data_dir
        self.splits_path = splits_path
        self.config_path = config_path
        self._dataset: CorpusRepairDataset | None = None
        self._index: BM25Index | None = None
        self._lock = Lock()

    def is_ready(self) -> bool:
        csv_path = self.data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv"
        if (
            not csv_path.is_file()
            or not self.splits_path.is_file()
            or not self.config_path.is_file()
        ):
            return False
        try:
            load_config(self.config_path)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def _initialize(self) -> tuple[CorpusRepairDataset, BM25Index]:
        if self._dataset is not None and self._index is not None:
            return self._dataset, self._index
        with self._lock:
            if self._dataset is not None and self._index is not None:
                return self._dataset, self._index
            if not self.is_ready():
                raise RetrievalUnavailable("required historical retrieval assets are unavailable")
            config = load_config(self.config_path)
            dataset = load_corpus_repair_dataset(
                self.data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv",
                self.splits_path,
                evidence_cutoff_year=config.evidence_cutoff_year,
                max_passage_chars=config.max_passage_chars,
                max_profile_passages=config.max_profile_passages,
                max_profile_identifier_chars=config.max_profile_identifier_chars,
                max_profile_context_chars=config.max_profile_context_chars,
                max_profile_chars=config.max_profile_chars,
            )
            index = BM25Index(
                tuple(context.text for context in dataset.contexts),
                k1=config.k1,
                b=config.b,
            )
            self._dataset = dataset
            self._index = index
            return dataset, index

    def retrieve(self, query_text: str, *, top_k: int) -> tuple[EvidenceItem, ...]:
        dataset, index = self._initialize()
        items = retrieve_passages(index, dataset, query_text, top_k=top_k)
        # Historical contexts may digest the full paragraph while displaying a bounded window.
        # Production integrity binds the exact application-displayed passage instead.
        return tuple(
            item.model_copy(
                update={"passage_digest": hashlib.sha256(item.passage.encode("utf-8")).hexdigest()}
            )
            for item in items
        )
