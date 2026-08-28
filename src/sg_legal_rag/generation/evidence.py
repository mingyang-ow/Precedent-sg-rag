from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sg_legal_rag.retrieval.benchmark import QueryRecord
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset, extract_context_window


class EvidenceCondition(StrEnum):
    ORACLE_GOLD = "oracle_gold_context"
    RETRIEVED = "retrieved_context"


class EvidenceOrigin(StrEnum):
    GOLD_QUERY_ROW = "gold_query_row"
    HISTORICAL_RETRIEVAL = "historical_retrieval"


class ExpectedAction(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    UNKNOWN_NEEDS_REVIEW = "unknown_needs_review"


class EvidenceSufficiencyBasis(StrEnum):
    GOLD_CITATION_RELATIONSHIP = "gold_query_row_citation_relationship"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    MANUAL_REVIEWED_SUFFICIENT = "manual_reviewed_sufficient"
    MANUAL_REVIEWED_INSUFFICIENT = "manual_reviewed_insufficient"


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
    origin: EvidenceOrigin
    gold_row_id: str | None
    citation_relationship_verified: bool

    @model_validator(mode="after")
    def validate_origin(self) -> EvidenceItem:
        if self.origin is EvidenceOrigin.GOLD_QUERY_ROW:
            if self.gold_row_id is None:
                raise ValueError("gold evidence requires a gold row ID")
            if self.retrieval_rank is not None or self.retrieval_score is not None:
                raise ValueError("gold evidence cannot carry retrieval rank or score")
        elif self.gold_row_id is not None:
            raise ValueError("historical retrieval evidence cannot carry a gold row ID")
        return self


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
    target_present: bool
    evidence_sufficient: bool | None
    expected_action: ExpectedAction
    sufficiency_basis: EvidenceSufficiencyBasis

    @model_validator(mode="after")
    def validate_methodology(self) -> EvidencePackage:
        supplied = {item.case_id for item in self.evidence}
        actual_target_present = bool(supplied & set(self.accepted_case_ids))
        if self.target_present != actual_target_present:
            raise ValueError("target_present must reflect supplied evidence case identity")
        expected_by_sufficiency = {
            True: ExpectedAction.ANSWER,
            False: ExpectedAction.ABSTAIN,
            None: ExpectedAction.UNKNOWN_NEEDS_REVIEW,
        }
        if self.expected_action is not expected_by_sufficiency[self.evidence_sufficient]:
            raise ValueError("expected_action must reflect evidence_sufficient")
        expected_basis = {
            True: {
                EvidenceSufficiencyBasis.GOLD_CITATION_RELATIONSHIP,
                EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT,
            },
            False: {EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT},
            None: {EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED},
        }
        if self.sufficiency_basis not in expected_basis[self.evidence_sufficient]:
            raise ValueError("sufficiency_basis must reflect evidence_sufficient")
        if self.condition is EvidenceCondition.ORACLE_GOLD:
            if self.top_k is not None:
                raise ValueError("oracle gold evidence cannot have top_k")
            if any(item.origin is not EvidenceOrigin.GOLD_QUERY_ROW for item in self.evidence):
                raise ValueError("oracle evidence must originate from a gold query row")
        else:
            if self.top_k is None:
                raise ValueError("retrieved evidence requires top_k")
            if any(
                item.origin is not EvidenceOrigin.HISTORICAL_RETRIEVAL for item in self.evidence
            ):
                raise ValueError("gold query-row evidence cannot enter retrieved context")
            if self.evidence_sufficient is not None and self.sufficiency_basis not in {
                EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT,
                EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT,
            }:
                raise ValueError("retrieved evidence sufficiency requires manual review")
        return self

    @property
    def answer_expected(self) -> bool:
        return self.expected_action is ExpectedAction.ANSWER


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
        origin=EvidenceOrigin.HISTORICAL_RETRIEVAL,
        gold_row_id=None,
        citation_relationship_verified=context.identifier_matched,
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
    target_present = bool(supplied & set(accepted))
    warm_start = any(
        value in dataset.historical_case_ids for value in relevant_case_ids(query, case_to_id)
    )
    return EvidencePackage(
        package_id=_package_id(identifier, EvidenceCondition.RETRIEVED, top_k),
        query_id=identifier,
        query_mode=mode,
        query_text=query.text,
        stratum=stratum,
        condition=EvidenceCondition.RETRIEVED,
        top_k=top_k,
        evidence=evidence,
        accepted_case_ids=accepted,
        warm_start=warm_start,
        target_present=target_present,
        evidence_sufficient=None,
        expected_action=ExpectedAction.UNKNOWN_NEEDS_REVIEW,
        sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED,
    )


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
    contexts = sorted(
        query.gold_contexts.values(),
        key=lambda item: (
            not item.identifier_matched,
            not bool(item.paragraph),
            item.case_key,
            item.row_id,
        ),
    )
    if not contexts:
        raise ValueError("oracle evidence requires a gold test-query row")
    context = contexts[0]
    if context.case_key not in query.relevant_texts:
        raise ValueError("gold row cited case is not a labelled query target")
    if mode == "facts_only" and context.fact_query != query.text:
        raise ValueError("facts-only oracle row does not match the query")
    if mode == "facts_principle" and f"{context.fact_query} {context.principle}" != query.text:
        raise ValueError("facts-plus-principle oracle row does not match the query")
    passage, window_matched = (
        extract_context_window(context.paragraph, context.raw_case, dataset.max_passage_chars)
        if context.paragraph
        else ("", False)
    )
    numeric_case_id = case_to_id[context.case_key]
    relationship_verified = bool(context.identifier_matched and window_matched)
    evidence = EvidenceItem(
        evidence_id="E1",
        case_id=case_label(numeric_case_id),
        case_name=dataset.case_texts[numeric_case_id],
        source_judgment=context.source_reference,
        source_url=context.source_url,
        source_year=context.source_year,
        passage=passage,
        passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        retrieval_rank=None,
        retrieval_score=None,
        origin=EvidenceOrigin.GOLD_QUERY_ROW,
        gold_row_id=context.row_id,
        citation_relationship_verified=relationship_verified,
    )
    accepted = tuple(case_label(value) for value in relevant)
    evidence_sufficient = True if relationship_verified else None
    return EvidencePackage(
        package_id=_package_id(identifier, EvidenceCondition.ORACLE_GOLD, None),
        query_id=identifier,
        query_mode=mode,
        query_text=query.text,
        stratum=stratum,
        condition=EvidenceCondition.ORACLE_GOLD,
        top_k=None,
        evidence=(evidence,),
        accepted_case_ids=accepted,
        warm_start=any(value in dataset.historical_case_ids for value in relevant),
        target_present=True,
        evidence_sufficient=evidence_sufficient,
        expected_action=(
            ExpectedAction.ANSWER if evidence_sufficient else ExpectedAction.UNKNOWN_NEEDS_REVIEW
        ),
        sufficiency_basis=(
            EvidenceSufficiencyBasis.GOLD_CITATION_RELATIONSHIP
            if evidence_sufficient
            else EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED
        ),
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
