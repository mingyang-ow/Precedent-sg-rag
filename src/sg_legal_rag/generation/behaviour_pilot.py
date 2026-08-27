from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adjudication import PilotAdjudication, adjudication_digest
from .evidence import (
    EvidenceCondition,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)

BEHAVIOUR_PILOT_VERSION = "pilot-behaviour-v1"
BEHAVIOUR_REVIEW_VERSION = "behaviour-sufficiency-v1"
RETRIEVED_ORDER_TAG = "retrieved"
ORACLE_FALLBACK_ORDER_TAG = "oracle-fallback"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_order(
    package_ids: tuple[str, ...],
    *,
    seed: int,
    tag: str,
    mode: str | None = None,
) -> tuple[str, ...]:
    prefix = f"{seed}\0{tag}\0" if mode is None else f"{seed}\0{tag}\0{mode}\0"
    return tuple(
        sorted(
            package_ids,
            key=lambda package_id: (
                hashlib.sha256(f"{prefix}{package_id}".encode()).hexdigest(),
                package_id,
            ),
        )
    )


class BehaviourReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    query_mode: Literal["facts_only", "facts_principle"]
    condition: EvidenceCondition
    top_k: int | None
    target_present: bool
    evidence_sufficient: bool
    expected_action: Literal[ExpectedAction.ANSWER, ExpectedAction.ABSTAIN]
    support_type: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...]
    borderline: bool
    review_rationale: str = Field(min_length=1)
    selected_for_pilot: bool
    review_order: int = Field(ge=1)
    reviewer: str = Field(min_length=1)
    review_date: date
    review_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_label(self) -> BehaviourReviewRecord:
        expected = ExpectedAction.ANSWER if self.evidence_sufficient else ExpectedAction.ABSTAIN
        if self.expected_action is not expected:
            raise ValueError("expected_action must reflect evidence_sufficient")
        if self.evidence_sufficient and not self.supporting_evidence_ids:
            raise ValueError("answer labels require supporting evidence IDs")
        if not self.evidence_sufficient and self.supporting_evidence_ids:
            raise ValueError("abstain labels cannot claim supporting evidence IDs")
        if self.condition is EvidenceCondition.ORACLE_GOLD and self.top_k is not None:
            raise ValueError("oracle review records cannot have top_k")
        if self.condition is EvidenceCondition.RETRIEVED and self.top_k is None:
            raise ValueError("retrieved review records require top_k")
        return self


class BehaviourAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    pilot_version: Literal["pilot-behaviour-v1"]
    adjudication_version: Literal["behaviour-sufficiency-v1"]
    blinded_to_model_outputs: Literal[True]
    reviewer: str = Field(min_length=1)
    review_date: date
    seed: int
    candidate_pool_definition: str = Field(min_length=1)
    candidate_order_algorithm: str = Field(min_length=1)
    candidate_pool_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_order_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_order: tuple[str, ...] = Field(min_length=1)
    oracle_initial_package_ids: tuple[str, str]
    oracle_fallback_order_algorithm: str = Field(min_length=1)
    reviewed_candidates: tuple[BehaviourReviewRecord, ...] = Field(min_length=1)
    oracle_reviews: tuple[BehaviourReviewRecord, ...] = Field(min_length=1)
    selected_package_ids: tuple[str, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def validate_review_metadata(self) -> BehaviourAdjudication:
        all_reviews = self.reviewed_candidates + self.oracle_reviews
        package_ids = [record.package_id for record in all_reviews]
        if len(set(package_ids)) != len(package_ids):
            raise ValueError("review package IDs must be unique")
        if any(record.reviewer != self.reviewer for record in all_reviews):
            raise ValueError("record reviewer must match adjudication reviewer")
        if any(record.review_date != self.review_date for record in all_reviews):
            raise ValueError("record review_date must match adjudication review_date")
        if any(record.review_version != self.adjudication_version for record in all_reviews):
            raise ValueError("record review_version must match adjudication_version")
        if len(set(self.selected_package_ids)) != 12:
            raise ValueError("selected package IDs must be 12 unique values")
        selected_from_reviews = {
            record.package_id for record in all_reviews if record.selected_for_pilot
        }
        if selected_from_reviews != set(self.selected_package_ids):
            raise ValueError("selected IDs must exactly match selected review records")
        return self


class BehaviourPilotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    pilot_version: Literal["pilot-behaviour-v1"]
    purpose: str = Field(min_length=1)
    prevalence_warning: str = Field(min_length=1)
    seed: int
    selected_package_ids: tuple[str, ...] = Field(min_length=12, max_length=12)
    adjudication_version: Literal["behaviour-sufficiency-v1"]
    adjudication_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest_algorithm: Literal["sha256-canonical-json-v1"]
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_manifest_schema_version: int
    global_run_signature: str
    global_evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_pilot_adjudication_version: str
    answer_pilot_adjudication_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_pilot_evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_contract: dict[str, Any]
    run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")
    counts: dict[str, dict[str, int]]
    estimate: dict[str, Any]
    metric_plan: dict[str, Any]


def load_behaviour_adjudication(path: Path) -> BehaviourAdjudication:
    return BehaviourAdjudication.model_validate_json(path.read_text(encoding="utf-8"))


def load_behaviour_pilot(path: Path) -> BehaviourPilotManifest:
    return BehaviourPilotManifest.model_validate_json(path.read_text(encoding="utf-8"))


def behaviour_adjudication_digest(adjudication: BehaviourAdjudication) -> str:
    return canonical_digest(adjudication.model_dump(mode="json"))


def behaviour_run_signature(
    *,
    global_run_signature: str,
    adjudication_digest_value: str,
    evidence_digest: str,
) -> str:
    return canonical_digest(
        {
            "cache_schema": 5,
            "pilot_version": BEHAVIOUR_PILOT_VERSION,
            "global_run_signature": global_run_signature,
            "adjudication_digest": adjudication_digest_value,
            "evidence_digest": evidence_digest,
        }
    )[:24]


def _expected_selected_records(
    reviews: tuple[BehaviourReviewRecord, ...],
) -> tuple[BehaviourReviewRecord, ...]:
    quotas = {
        ("facts_only", ExpectedAction.ANSWER): 2,
        ("facts_only", ExpectedAction.ABSTAIN): 3,
        ("facts_principle", ExpectedAction.ANSWER): 2,
        ("facts_principle", ExpectedAction.ABSTAIN): 3,
    }
    selected: list[BehaviourReviewRecord] = []
    for record in sorted(reviews, key=lambda item: item.review_order):
        key = (record.query_mode, ExpectedAction(record.expected_action))
        if (
            sum(item.query_mode == key[0] and item.expected_action is key[1] for item in selected)
            < quotas[key]
        ):
            selected.append(record)
    if Counter((item.query_mode, item.expected_action) for item in selected) != Counter(quotas):
        raise ValueError("reviewed prefix did not fill all retrieved quotas")
    return tuple(selected)


def apply_behaviour_labels(
    package: EvidencePackage,
    records_by_package: dict[str, BehaviourReviewRecord],
) -> EvidencePackage:
    record = records_by_package.get(package.package_id)
    if record is None:
        return package
    payload = package.model_dump(mode="python")
    payload.update(
        evidence_sufficient=record.evidence_sufficient,
        expected_action=record.expected_action,
        sufficiency_basis=(
            EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT
            if record.evidence_sufficient
            else EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT
        ),
    )
    return EvidencePackage.model_validate(payload)


def validate_behaviour_pilot(
    *,
    adjudication: BehaviourAdjudication,
    pilot: BehaviourPilotManifest,
    packages: tuple[EvidencePackage, ...],
    global_manifest: dict[str, Any],
    answer_adjudication: PilotAdjudication,
) -> tuple[tuple[EvidencePackage, ...], tuple[EvidencePackage, ...]]:
    if pilot.pilot_version != adjudication.pilot_version:
        raise ValueError("behaviour pilot version mismatch")
    if pilot.seed != adjudication.seed:
        raise ValueError("behaviour pilot seed mismatch")
    digest = behaviour_adjudication_digest(adjudication)
    if pilot.adjudication_digest != digest:
        raise ValueError("behaviour adjudication digest mismatch")

    package_by_id = {package.package_id: package for package in packages}
    old_reviewed = {record.package_id for record in answer_adjudication.records}
    candidate_pool = tuple(
        package.package_id
        for package in packages
        if package.condition is EvidenceCondition.RETRIEVED
        and package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
        and package.package_id not in old_reviewed
    )
    candidate_order = deterministic_order(
        candidate_pool,
        seed=adjudication.seed,
        tag=RETRIEVED_ORDER_TAG,
    )
    if canonical_digest(candidate_pool) != adjudication.candidate_pool_signature:
        raise ValueError("behaviour candidate-pool signature mismatch")
    if candidate_order != adjudication.candidate_order:
        raise ValueError("behaviour candidate order changed")
    if canonical_digest(candidate_order) != adjudication.candidate_order_signature:
        raise ValueError("behaviour candidate-order signature mismatch")

    reviewed = tuple(sorted(adjudication.reviewed_candidates, key=lambda item: item.review_order))
    if tuple(record.review_order for record in reviewed) != tuple(range(1, len(reviewed) + 1)):
        raise ValueError("retrieved review order must be consecutive")
    if tuple(record.package_id for record in reviewed) != candidate_order[: len(reviewed)]:
        raise ValueError("reviewed candidates must be an exact prefix of the blind order")
    expected_selected = _expected_selected_records(reviewed)
    if {item.package_id for item in expected_selected} != {
        item.package_id for item in reviewed if item.selected_for_pilot
    }:
        raise ValueError("retrieved selections do not follow first-observed quota filling")

    for record in reviewed + adjudication.oracle_reviews:
        package = package_by_id.get(record.package_id)
        if package is None:
            raise ValueError(
                f"review package is absent from frozen evaluation: {record.package_id}"
            )
        if (
            package.query_mode != record.query_mode
            or package.condition is not record.condition
            or package.top_k != record.top_k
            or package.target_present != record.target_present
        ):
            raise ValueError(f"review metadata changed: {record.package_id}")
        evidence_ids = {item.evidence_id for item in package.evidence}
        if not set(record.supporting_evidence_ids).issubset(evidence_ids):
            raise ValueError(f"review cites unknown evidence: {record.package_id}")

    oracle_selected = tuple(
        record for record in adjudication.oracle_reviews if record.selected_for_pilot
    )
    oracle_by_id = {record.package_id: record for record in adjudication.oracle_reviews}
    if any(
        package_id not in oracle_by_id for package_id in adjudication.oracle_initial_package_ids
    ):
        raise ValueError("initial oracle selections must be present in the oracle review log")
    facts_only_fallback_reviews = tuple(
        record
        for record in sorted(
            adjudication.oracle_reviews,
            key=lambda item: item.review_order,
        )
        if record.query_mode == "facts_only"
        and record.package_id not in adjudication.oracle_initial_package_ids
    )
    oracle_fallback_pool = tuple(
        package.package_id
        for package in packages
        if package.condition is EvidenceCondition.ORACLE_GOLD
        and package.query_mode == "facts_only"
        and package.package_id not in answer_adjudication.pilot_package_ids
        and package.package_id not in adjudication.oracle_initial_package_ids
    )
    expected_oracle_fallback_order = deterministic_order(
        oracle_fallback_pool,
        seed=adjudication.seed,
        tag=ORACLE_FALLBACK_ORDER_TAG,
        mode="facts_only",
    )
    if (
        tuple(record.package_id for record in facts_only_fallback_reviews)
        != (expected_oracle_fallback_order[: len(facts_only_fallback_reviews)])
    ):
        raise ValueError("oracle fallback reviews do not follow the frozen order")
    if Counter(record.query_mode for record in oracle_selected) != {
        "facts_only": 1,
        "facts_principle": 1,
    } or any(record.expected_action is not ExpectedAction.ANSWER for record in oracle_selected):
        raise ValueError("behaviour pilot requires one answerable oracle per query mode")

    selected_ids = adjudication.selected_package_ids
    if pilot.selected_package_ids != selected_ids:
        raise ValueError("behaviour selected package ordering changed")
    raw_selected = tuple(package_by_id[package_id] for package_id in selected_ids)
    records_by_package = {
        record.package_id: record for record in reviewed + adjudication.oracle_reviews
    }
    labeled_selected = tuple(
        apply_behaviour_labels(package, records_by_package) for package in raw_selected
    )
    if any(
        package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
        for package in labeled_selected
    ):
        raise ValueError("behaviour pilot contains unresolved ground truth")

    counts = {
        "expected_actions": dict(
            sorted(Counter(package.expected_action.value for package in labeled_selected).items())
        ),
        "query_modes": dict(
            sorted(Counter(package.query_mode for package in labeled_selected).items())
        ),
        "conditions": dict(
            sorted(Counter(package.condition.value for package in labeled_selected).items())
        ),
        "top_k": dict(
            sorted(
                Counter(
                    "oracle" if package.top_k is None else str(package.top_k)
                    for package in labeled_selected
                ).items()
            )
        ),
    }
    if counts != pilot.counts:
        raise ValueError("behaviour pilot counts changed")
    if counts["expected_actions"] != {"abstain": 6, "answer": 6}:
        raise ValueError("behaviour pilot must contain six answer and six abstain records")
    if counts["query_modes"] != {"facts_only": 6, "facts_principle": 6}:
        raise ValueError("behaviour pilot must contain six records per query mode")
    if counts["conditions"] != {"oracle_gold_context": 2, "retrieved_context": 10}:
        raise ValueError("behaviour pilot condition mix changed")

    frozen_locks = global_manifest.get("evidence_freeze", {}).get("packages")
    if not isinstance(frozen_locks, list):
        raise TypeError("global manifest lacks frozen package locks")
    locks_by_id = {lock["package_id"]: lock for lock in frozen_locks}
    selected_locks = [locks_by_id[package_id] for package_id in selected_ids]
    if canonical_digest(selected_locks) != pilot.evidence_digest:
        raise ValueError("behaviour pilot evidence digest mismatch")
    if global_manifest["evidence_freeze"]["signature"] != pilot.global_evidence_digest:
        raise ValueError("global evidence digest changed")
    if global_manifest["run_signature"] != pilot.global_run_signature:
        raise ValueError("global run signature changed")
    if global_manifest["generation_contract"] != pilot.generation_contract:
        raise ValueError("behaviour generation contract changed")
    if adjudication_digest(answer_adjudication) != pilot.answer_pilot_adjudication_digest:
        raise ValueError("answer-only pilot adjudication changed")
    if answer_adjudication.pilot_evidence_signature != pilot.answer_pilot_evidence_digest:
        raise ValueError("answer-only pilot evidence changed")
    expected_run_signature = behaviour_run_signature(
        global_run_signature=pilot.global_run_signature,
        adjudication_digest_value=digest,
        evidence_digest=pilot.evidence_digest,
    )
    if pilot.run_signature != expected_run_signature:
        raise ValueError("behaviour pilot run signature changed")
    return raw_selected, labeled_selected
