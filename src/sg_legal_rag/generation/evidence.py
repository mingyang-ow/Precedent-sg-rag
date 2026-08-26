from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from sg_legal_rag.retrieval.benchmark import QueryRecord
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset


class EvidenceCondition(StrEnum):
    ORACLE = "oracle_context"
    RETRIEVED = "retrieved_context"
    INSUFFICIENT = "insufficient_evidence"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    case_id: str = Field(pattern=r"^case:[0-9]+$")
    case_name: str
    source_judgment: str
    source_url: str
    source_year: int
    passage: str
    passage_digest: str
    retrieval_rank: int | None
    retrieval_score: float | None


class EvidencePackage(BaseModel):
    """Complete experiment input; evaluation-only labels never enter the prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    query_id: str
    query_mode: str
    query_text: str
    stratum: str
    condition: EvidenceCondition
    top_k: int | None
    evidence: tuple[EvidenceItem, ...]
    accepted_case_ids: tuple[str, ...]
    warm_start: bool
    retrieval_correct: bool

    @property
    def answer_expected(self) -> bool:
        return self.condition is not EvidenceCondition.INSUFFICIENT


def case_label(case_id: int) -> str:
    if case_id < 0:
        raise ValueError("case ID must be non-negative")
    return f"case:{case_id}"


def query_id(mode: str, query: QueryRecord) -> str:
    payload = json.dumps(
        [mode, query.text, sorted(query.relevant_texts)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _package_id(query_identifier: str, condition: EvidenceCondition, top_k: int | None) -> str:
    payload = f"{query_identifier}\0{condition.value}\0{top_k}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _item(
    dataset: CorpusRepairDataset,
    passage_id: int,
    evidence_number: int,
    *,
    rank: int | None,
    score: float | None,
) -> EvidenceItem:
    context = dataset.contexts[passage_id]
    numeric_case_id = int(dataset.context_case_ids[passage_id])
    return EvidenceItem(
        evidence_id=f"E{evidence_number}",
        case_id=case_label(numeric_case_id),
        case_name=dataset.case_texts[numeric_case_id],
        source_judgment=context.source_reference,
        source_url=context.source_url,
        source_year=context.source_year,
        passage=context.text,
        passage_digest=context.digest,
        retrieval_rank=rank,
        retrieval_score=score,
    )


def retrieve_passages(
    index: BM25Index,
    dataset: CorpusRepairDataset,
    query_text: str,
    *,
    top_k: int,
) -> tuple[EvidenceItem, ...]:
    """Return the highest-scoring positive passage for each top-ranked case."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    passage_scores = index.scores(query_text)
    best_by_case: dict[int, tuple[float, int]] = {}
    for passage_id, score in passage_scores.items():
        numeric_case_id = int(dataset.context_case_ids[passage_id])
        previous = best_by_case.get(numeric_case_id)
        candidate = (float(score), passage_id)
        if (
            previous is None
            or score > previous[0]
            or (score == previous[0] and passage_id < previous[1])
        ):
            best_by_case[numeric_case_id] = candidate
    ranked = sorted(
        best_by_case.items(),
        key=lambda pair: (-pair[1][0], pair[0], pair[1][1]),
    )[:top_k]
    return tuple(
        _item(dataset, passage_id, rank, rank=rank, score=score)
        for rank, (_, (score, passage_id)) in enumerate(ranked, start=1)
    )


def relevant_case_ids(query: QueryRecord, case_to_id: dict[str, int]) -> tuple[int, ...]:
    return tuple(sorted(case_to_id[target] for target in query.relevant_texts))


def package_retrieved_evidence(
    *,
    mode: str,
    query: QueryRecord,
    stratum: str,
    top_k: int,
    index: BM25Index,
    dataset: CorpusRepairDataset,
    case_to_id: dict[str, int],
) -> EvidencePackage:
    identifier = query_id(mode, query)
    evidence = retrieve_passages(index, dataset, query.text, top_k=top_k)
    accepted = tuple(case_label(value) for value in relevant_case_ids(query, case_to_id))
    supplied = {item.case_id for item in evidence}
    retrieval_correct = bool(supplied & set(accepted))
    condition = EvidenceCondition.RETRIEVED if retrieval_correct else EvidenceCondition.INSUFFICIENT
    warm_start = any(
        value in dataset.historical_case_ids for value in relevant_case_ids(query, case_to_id)
    )
    return EvidencePackage(
        package_id=_package_id(identifier, condition, top_k),
        query_id=identifier,
        query_mode=mode,
        query_text=query.text,
        stratum=stratum,
        condition=condition,
        top_k=top_k,
        evidence=evidence,
        accepted_case_ids=accepted,
        warm_start=warm_start,
        retrieval_correct=retrieval_correct,
    )


def _contexts_by_case(dataset: CorpusRepairDataset) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for passage_id, numeric_case_id in enumerate(dataset.context_case_ids):
        grouped[int(numeric_case_id)].append(passage_id)
    return grouped


def package_oracle_evidence(
    *,
    mode: str,
    query: QueryRecord,
    stratum: str,
    dataset: CorpusRepairDataset,
    case_to_id: dict[str, int],
) -> EvidencePackage:
    identifier = query_id(mode, query)
    relevant = relevant_case_ids(query, case_to_id)
    grouped = _contexts_by_case(dataset)
    oracle_case = next((case_id for case_id in relevant if grouped.get(case_id)), None)
    if oracle_case is None:
        raise ValueError("oracle evidence requires at least one warm relevant case")
    passage_id = grouped[oracle_case][0]
    accepted = tuple(case_label(value) for value in relevant)
    return EvidencePackage(
        package_id=_package_id(identifier, EvidenceCondition.ORACLE, None),
        query_id=identifier,
        query_mode=mode,
        query_text=query.text,
        stratum=stratum,
        condition=EvidenceCondition.ORACLE,
        top_k=None,
        evidence=(_item(dataset, passage_id, 1, rank=None, score=None),),
        accepted_case_ids=accepted,
        warm_start=True,
        retrieval_correct=True,
    )


def prompt_evidence(package: EvidencePackage) -> list[dict[str, object]]:
    """Serialize only model-visible evidence, excluding evaluation labels."""

    return [
        {
            "evidence_id": item.evidence_id,
            "case_id": item.case_id,
            "case_name": item.case_name,
            "source_judgment": item.source_judgment,
            "source_year": item.source_year,
            "passage": item.passage,
        }
        for item in package.evidence
    ]
