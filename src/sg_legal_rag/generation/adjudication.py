from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import (
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)


class PilotAdjudicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    target_present: bool
    evidence_sufficient: bool
    expected_action: ExpectedAction
    review_rationale: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    support_granularity: Literal["exact_passage", "multiple_supplied_passages"]
    passage_support_note: str = Field(min_length=1)
    borderline: bool
    reviewer: str = Field(min_length=1)
    review_date: date
    review_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_expected_action(self) -> PilotAdjudicationRecord:
        expected = ExpectedAction.ANSWER if self.evidence_sufficient else ExpectedAction.ABSTAIN
        if self.expected_action is not expected:
            raise ValueError("expected_action must reflect evidence_sufficient")
        return self


class PilotAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    adjudication_version: str = Field(min_length=1)
    blinded_to_model_outputs: Literal[True]
    reviewer: str = Field(min_length=1)
    review_date: date
    pilot_package_ids: tuple[str, ...] = Field(min_length=1)
    pilot_evidence_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_pilot_evidence_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[PilotAdjudicationRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review_metadata(self) -> PilotAdjudication:
        package_ids = [record.package_id for record in self.records]
        if len(set(package_ids)) != len(package_ids):
            raise ValueError("adjudication package IDs must be unique")
        if any(record.reviewer != self.reviewer for record in self.records):
            raise ValueError("record reviewer must match adjudication reviewer")
        if any(record.review_date != self.review_date for record in self.records):
            raise ValueError("record review_date must match adjudication review_date")
        if any(record.review_version != self.adjudication_version for record in self.records):
            raise ValueError("record review_version must match adjudication_version")
        return self


def load_pilot_adjudication(path: Path) -> PilotAdjudication:
    return PilotAdjudication.model_validate_json(path.read_text(encoding="utf-8"))


def adjudication_digest(adjudication: PilotAdjudication) -> str:
    payload = json.dumps(
        adjudication.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_adjudication(
    package: EvidencePackage,
    records_by_package: dict[str, PilotAdjudicationRecord],
) -> EvidencePackage:
    record = records_by_package.get(package.package_id)
    if record is None:
        return package
    if package.target_present != record.target_present:
        raise ValueError(f"adjudication target_present mismatch: {package.package_id}")
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
