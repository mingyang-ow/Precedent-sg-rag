from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from sg_legal_rag.api.app import create_app
from sg_legal_rag.api.retrieval import PreparedPassageBM25Retriever, RetrievalUnavailable
from sg_legal_rag.api.settings import ApiSettings
from sg_legal_rag.generation.evidence import retrieve_passages
from sg_legal_rag.retrieval.artifacts import (
    ARTIFACT_VERSION,
    BM25_FILES,
    CORPUS_FILE,
    MANIFEST_FILE,
    RetrievalArtifactError,
    RetrievalBuildProvenance,
    load_retrieval_artifacts,
    write_retrieval_bundle,
)
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset, HistoricalContext


def _context(case_name: str, text: str, position: int) -> HistoricalContext:
    return HistoricalContext(
        case_key=case_name.casefold(),
        raw_case=case_name,
        source_url=f"https://example.test/judgment-{position}",
        source_reference=f"[2023] SGHC {position}",
        source_year=2023,
        text=text,
        original_chars=len(text),
        identifier_matched=True,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@pytest.fixture
def corpus() -> CorpusRepairDataset:
    cases = (
        "Alpha v Beta [2020] SGCA 2",
        "Crown v Delta [2019] SGHC 4",
        "Echo v Foxtrot [2018] SGCA 6",
    )
    contexts = (
        _context(cases[0], "Alpha v Beta applied objective contract interpretation.", 1),
        _context(cases[1], "Crown v Delta addressed proportional criminal sentencing.", 2),
        _context(cases[0], "Alpha v Beta also considered expectation damages.", 3),
        _context(cases[2], "Echo v Foxtrot concerned a fiduciary duty.", 4),
    )
    return CorpusRepairDataset(
        case_keys=tuple(value.casefold() for value in cases),
        case_texts=cases,
        contexts=contexts,
        context_case_ids=np.asarray([0, 1, 0, 2], dtype=np.int64),
        profiles=cases,
        historical_case_ids=frozenset({0, 1, 2}),
        queries_by_mode={},
        audit={},
        test_urls=frozenset(),
        max_passage_chars=4000,
    )


@pytest.fixture
def provenance() -> RetrievalBuildProvenance:
    return RetrievalBuildProvenance(
        dataset_id="fixture/legal-citations",
        dataset_revision="revision-1",
        source_file="fixture.csv",
        source_digest="1" * 64,
        source_size=1234,
        config_digest="2" * 64,
        splits_digest="3" * 64,
        source_date_epoch=1_700_000_000,
    )


def _bundle(
    path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
) -> tuple[Path, BM25Index]:
    index = BM25Index([context.text for context in corpus.contexts], k1=1.2, b=0.75)
    write_retrieval_bundle(
        dataset=corpus,
        index=index,
        provenance=provenance,
        output_dir=path,
    )
    return path, index


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class OfflineASGIClient:
    def __init__(self, application: Any) -> None:
        self.application = application

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            async with (
                self.application.router.lifespan_context(self.application),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=self.application),
                    base_url="http://testserver",
                ) as client,
            ):
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str) -> httpx.Response:
        return self.request("GET", path)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)


def test_bundle_and_manifest_are_deterministic(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
) -> None:
    first, _ = _bundle(tmp_path / "first", corpus, provenance)
    second, _ = _bundle(tmp_path / "second", corpus, provenance)

    for file_name in (MANIFEST_FILE, CORPUS_FILE, *BM25_FILES.values()):
        assert (first / file_name).read_bytes() == (second / file_name).read_bytes()


def test_loaded_retrieval_is_canonically_equivalent(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
) -> None:
    path, canonical_index = _bundle(tmp_path / "bundle", corpus, provenance)
    loaded = load_retrieval_artifacts(path)

    assert loaded.corpus.context_case_ids.flags.writeable is False
    assert loaded.index.document_lengths.flags.writeable is False
    assert loaded.index.posting_document_ids.flags.writeable is False

    for query in (
        "objective contract interpretation",
        "criminal sentencing",
        "fiduciary duty",
        "expectation damages",
    ):
        canonical = retrieve_passages(canonical_index, corpus, query, top_k=3)
        restored = retrieve_passages(loaded.index, loaded.corpus, query, top_k=3)
        assert [item.case_id for item in restored] == [item.case_id for item in canonical]
        assert [item.evidence_id for item in restored] == [item.evidence_id for item in canonical]
        assert [item.retrieval_rank for item in restored] == [
            item.retrieval_rank for item in canonical
        ]
        assert [item.retrieval_score for item in restored] == pytest.approx(
            [item.retrieval_score for item in canonical], rel=0, abs=0
        )


def test_corrupted_artifact_is_rejected(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
) -> None:
    path, _ = _bundle(tmp_path / "bundle", corpus, provenance)
    content = bytearray((path / CORPUS_FILE).read_bytes())
    content[-1] ^= 1
    (path / CORPUS_FILE).write_bytes(content)

    with pytest.raises(RetrievalArtifactError, match="file integrity"):
        load_retrieval_artifacts(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (None, "artifact_schema_version", 99, "schema_version is incompatible"),
        ("bm25", "k1", 2.0, "BM25 parameters are incompatible"),
    ],
)
def test_incompatible_version_or_configuration_is_rejected(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
    section: str | None,
    field: str,
    value: object,
    message: str,
) -> None:
    path, _ = _bundle(tmp_path / "bundle", corpus, provenance)
    manifest = json.loads((path / MANIFEST_FILE).read_bytes())
    target = manifest if section is None else manifest[section]
    target[field] = value
    (path / MANIFEST_FILE).write_bytes(_canonical(manifest))

    with pytest.raises(RetrievalArtifactError, match=message):
        load_retrieval_artifacts(path)


def test_missing_bundle_is_unavailable_without_rebuild(tmp_path: Path) -> None:
    retriever = PreparedPassageBM25Retriever(artifact_dir=tmp_path / "missing")

    assert retriever.is_ready() is False
    with pytest.raises(RetrievalUnavailable):
        retriever.retrieve("query", top_k=2)


def test_prepared_retriever_never_reconstructs_state_during_serving(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sg_legal_rag.retrieval.artifacts as artifact_module

    path, _ = _bundle(tmp_path / "bundle", corpus, provenance)

    def forbidden_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("normal serving must not reconstruct retrieval state")

    monkeypatch.setattr(artifact_module, "load_corpus_repair_dataset", forbidden_build)
    monkeypatch.setattr(BM25Index, "__init__", forbidden_build)
    retriever = PreparedPassageBM25Retriever(artifact_dir=path)
    results = retriever.retrieve("objective contract", top_k=2)

    assert results[0].case_id == "case:0"


def test_api_uses_prepared_bundle_and_exposes_safe_identity(
    tmp_path: Path,
    corpus: CorpusRepairDataset,
    provenance: RetrievalBuildProvenance,
) -> None:
    path, _ = _bundle(tmp_path / "bundle", corpus, provenance)
    client = OfflineASGIClient(
        create_app(settings=ApiSettings(retrieval_artifact_dir=path, openai_api_key=None))
    )

    health = client.get("/health")
    ready = client.get("/ready")
    version = client.get("/version")
    retrieved = client.post("/retrieve", json={"facts": "objective contract", "top_k": 2})

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "partial"
    assert version.json()["retrieval_artifact_version"] == ARTIFACT_VERSION
    assert len(version.json()["retrieval_artifact_digest"]) == 64
    assert "bundle" not in version.text
    assert retrieved.status_code == 200
    assert retrieved.json()["results"][0]["case_id"] == "case:0"


def test_api_stays_live_but_not_ready_for_invalid_bundle(tmp_path: Path) -> None:
    client = OfflineASGIClient(
        create_app(
            settings=ApiSettings(
                retrieval_artifact_dir=tmp_path / "missing",
                openai_api_key=None,
            )
        )
    )

    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
    assert client.post("/retrieve", json={"facts": "query"}).status_code == 503
