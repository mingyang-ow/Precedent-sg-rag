from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sg_legal_rag.generation.evidence import EvidenceItem, retrieve_passages
from sg_legal_rag.retrieval.artifacts import (
    LoadedRetrievalArtifacts,
    RetrievalArtifactError,
    RetrievalArtifactIdentity,
    load_retrieval_artifacts,
)

from .observability import log_event


class EvidenceRetriever(Protocol):
    def is_ready(self) -> bool: ...

    def retrieve(self, query_text: str, *, top_k: int) -> tuple[EvidenceItem, ...]: ...


class RetrievalUnavailable(RuntimeError):
    pass


class PreparedPassageBM25Retriever:
    """Verify and load immutable passage-BM25 state during service construction."""

    def __init__(self, *, artifact_dir: Path) -> None:
        self._loaded: LoadedRetrievalArtifacts | None = None
        self._failure: str | None = None
        try:
            self._loaded = load_retrieval_artifacts(artifact_dir)
        except (OSError, RetrievalArtifactError, TypeError, ValueError) as error:
            self._failure = str(error)
            log_event("retrieval_artifacts_unavailable", artifact_failure=self._failure)
        else:
            log_event(
                "retrieval_artifacts_loaded",
                artifact_digest=self._loaded.identity.manifest_digest,
                artifact_documents=self._loaded.identity.document_count,
                artifact_load_ms=round(self._loaded.identity.load_ms, 3),
            )

    def is_ready(self) -> bool:
        return self._loaded is not None

    def artifact_identity(self) -> RetrievalArtifactIdentity | None:
        return self._loaded.identity if self._loaded is not None else None

    def retrieve(self, query_text: str, *, top_k: int) -> tuple[EvidenceItem, ...]:
        if self._loaded is None:
            raise RetrievalUnavailable("prepared retrieval artifacts are unavailable")
        return retrieve_passages(
            self._loaded.index,
            self._loaded.corpus,
            query_text,
            top_k=top_k,
        )
