from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import EvidencePackage, prompt_evidence
from .provider import SYSTEM_INSTRUCTIONS, render_user_input

CLEANROOM_ADJUDICATION_VERSION = "behaviour-cleanroom-v1"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CleanroomLabel(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    BORDERLINE = "borderline"
    CANNOT_DETERMINE = "cannot_determine"


class CleanroomReviewRecord(BaseModel):
    """Only fields present in the exact rendered model input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_number: int = Field(ge=1, le=12)
    package_id: str
    query_mode: str
    query: str
    evidence: tuple[dict[str, object], ...] = Field(min_length=1)
    rendered_input_signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class CleanroomReviewArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    purpose: Literal["blind_model_visible_behaviour_adjudication"]
    adjudication_rule: str
    records: tuple[CleanroomReviewRecord, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def validate_records(self) -> CleanroomReviewArtifact:
        if tuple(record.record_number for record in self.records) != tuple(range(1, 13)):
            raise ValueError("clean-room record numbers must be consecutive")
        package_ids = tuple(record.package_id for record in self.records)
        if len(set(package_ids)) != 12:
            raise ValueError("clean-room review requires 12 unique packages")
        return self


class CleanroomAdjudicationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_number: int = Field(ge=1, le=12)
    package_id: str
    label: CleanroomLabel
    rationale: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_support(self) -> CleanroomAdjudicationRecord:
        if self.label is CleanroomLabel.ANSWER and not self.supporting_evidence_ids:
            raise ValueError("answer labels require supporting evidence IDs")
        if self.label is not CleanroomLabel.ANSWER and self.supporting_evidence_ids:
            raise ValueError("only answer labels may claim supporting evidence IDs")
        return self


class CleanroomAdjudication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    adjudication_version: Literal["behaviour-cleanroom-v1"]
    source_run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")
    source_evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    blinded_to_model_outputs: Literal[True]
    hidden_fields: tuple[str, ...]
    reviewer: str = Field(min_length=1)
    review_date: date
    records: tuple[CleanroomAdjudicationRecord, ...] = Field(min_length=12, max_length=12)
    digest_algorithm: Literal["sha256-canonical-json-v1"]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_adjudication(self) -> CleanroomAdjudication:
        if tuple(record.record_number for record in self.records) != tuple(range(1, 13)):
            raise ValueError("clean-room adjudication order changed")
        if len({record.package_id for record in self.records}) != 12:
            raise ValueError("clean-room adjudication requires 12 unique packages")
        if cleanroom_adjudication_digest(self) != self.digest:
            raise ValueError("clean-room adjudication digest mismatch")
        return self


def build_cleanroom_review(
    packages: tuple[EvidencePackage, ...],
) -> CleanroomReviewArtifact:
    records = tuple(
        CleanroomReviewRecord(
            record_number=position,
            package_id=package.package_id,
            query_mode=package.query_mode,
            query=package.query_text,
            evidence=tuple(prompt_evidence(package)),
            rendered_input_signature=hashlib.sha256(
                render_user_input(package).encode("utf-8")
            ).hexdigest(),
        )
        for position, package in enumerate(packages, start=1)
    )
    return CleanroomReviewArtifact(
        schema_version=1,
        purpose="blind_model_visible_behaviour_adjudication",
        adjudication_rule=SYSTEM_INSTRUCTIONS,
        records=records,
    )


def cleanroom_review_digest(artifact: CleanroomReviewArtifact) -> str:
    return canonical_digest(artifact.model_dump(mode="json"))


def cleanroom_adjudication_digest(adjudication: CleanroomAdjudication) -> str:
    return canonical_digest(adjudication.model_dump(mode="json", exclude={"digest"}))


def load_cleanroom_adjudication(path: Path) -> CleanroomAdjudication:
    return CleanroomAdjudication.model_validate_json(path.read_text(encoding="utf-8"))


def render_cleanroom_review(artifact: CleanroomReviewArtifact) -> str:
    lines = [
        "# Blind Behavioral Pilot Review",
        "",
        "Review only the material below. Do not inspect repository files or prior results.",
        "For each record, assign exactly one label: `answer`, `abstain`, `borderline`, or ",
        "`cannot_determine`, and give a concise rationale based only on the visible material.",
        "",
        "## Adjudication rule",
        "",
        artifact.adjudication_rule,
        "",
    ]
    for record in artifact.records:
        lines.extend(
            [
                f"## Record {record.record_number}",
                "",
                f"- Package ID: `{record.package_id}`",
                f"- Query mode: `{record.query_mode}`",
                "",
                "### Query",
                "",
                record.query,
                "",
                "### Evidence (exact order shown to the model)",
                "",
                "```json",
                json.dumps(record.evidence, indent=2, ensure_ascii=False),
                "```",
                "",
                "### Reviewer decision",
                "",
                "- Label:",
                "- Rationale:",
                "",
            ]
        )
    return "\n".join(lines)
