from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sg_legal_rag.retrieval.benchmark import QueryRecord
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset

from .evidence import (
    EvidencePackage,
    package_oracle_evidence,
    package_retrieved_evidence,
    query_id,
    relevant_case_ids,
    retrieve_passages,
)

WARM_SUCCESS = "warm_retrieval_success"
WARM_FAILURE = "warm_retrieval_failure"
COLD = "cold_start"
STRATA = (WARM_SUCCESS, WARM_FAILURE, COLD)


@dataclass(frozen=True)
class SelectedQuery:
    mode: str
    query: QueryRecord
    query_id: str
    stratum: str


def _selection_key(seed: int, mode: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{mode}\0{identifier}".encode()).hexdigest()


def classify_query(
    mode: str,
    query: QueryRecord,
    *,
    index: BM25Index,
    dataset: CorpusRepairDataset,
    case_to_id: dict[str, int],
    retrieval_depth: int,
) -> str:
    relevant = set(relevant_case_ids(query, case_to_id))
    warm = bool(relevant & dataset.historical_case_ids)
    if not warm:
        return COLD
    retrieved = retrieve_passages(index, dataset, query.text, top_k=retrieval_depth)
    selected = {item.case_id for item in retrieved}
    accepted = {f"case:{value}" for value in relevant}
    return WARM_SUCCESS if selected & accepted else WARM_FAILURE


def select_queries(
    *,
    modes: tuple[str, ...],
    dataset: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    per_stratum: int,
    retrieval_depth: int,
    seed: int,
) -> tuple[SelectedQuery, ...]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if retrieval_depth < 1:
        raise ValueError("retrieval_depth must be positive")

    selected: list[SelectedQuery] = []
    for mode in modes:
        if mode not in dataset.queries_by_mode:
            raise ValueError(f"unknown query mode: {mode}")
        buckets: dict[str, list[SelectedQuery]] = {stratum: [] for stratum in STRATA}
        candidates = sorted(
            ((query_id(mode, query), query) for query in dataset.queries_by_mode[mode]),
            key=lambda item: (_selection_key(seed, mode, item[0]), item[0]),
        )
        for identifier, query in candidates:
            stratum = classify_query(
                mode,
                query,
                index=index,
                dataset=dataset,
                case_to_id=case_to_id,
                retrieval_depth=retrieval_depth,
            )
            if len(buckets[stratum]) < per_stratum:
                buckets[stratum].append(SelectedQuery(mode, query, identifier, stratum))
            if all(len(items) == per_stratum for items in buckets.values()):
                break
        for stratum in STRATA:
            if len(buckets[stratum]) < per_stratum:
                raise ValueError(
                    f"{mode}/{stratum} has {len(buckets[stratum])} queries; {per_stratum} required"
                )
            selected.extend(buckets[stratum])
    return tuple(selected)


def build_packages(
    selected: tuple[SelectedQuery, ...],
    *,
    top_ks: tuple[int, ...],
    index: BM25Index,
    dataset: CorpusRepairDataset,
    case_to_id: dict[str, int],
) -> tuple[EvidencePackage, ...]:
    if not top_ks or any(value < 1 for value in top_ks):
        raise ValueError("top_ks must be non-empty and positive")
    if tuple(sorted(set(top_ks))) != top_ks:
        raise ValueError("top_ks must be unique and increasing")
    packages: list[EvidencePackage] = []
    for item in selected:
        if item.stratum != COLD:
            packages.append(
                package_oracle_evidence(
                    mode=item.mode,
                    query=item.query,
                    stratum=item.stratum,
                    dataset=dataset,
                    case_to_id=case_to_id,
                )
            )
        for top_k in top_ks:
            packages.append(
                package_retrieved_evidence(
                    mode=item.mode,
                    query=item.query,
                    stratum=item.stratum,
                    top_k=top_k,
                    index=index,
                    dataset=dataset,
                    case_to_id=case_to_id,
                )
            )
    if len({package.package_id for package in packages}) != len(packages):
        raise ValueError("package identifiers are not unique")
    return tuple(packages)
