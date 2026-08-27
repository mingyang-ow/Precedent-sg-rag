from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark import (
    DEFAULT_BEHAVIOUR_ADJUDICATION,
    DEFAULT_BEHAVIOUR_PACKAGES,
    DEFAULT_BEHAVIOUR_PILOT,
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_PILOT_ADJUDICATION,
    PROJECT_ROOT,
    preflight_frozen_behaviour_execution,
    write_json,
)
from .citation_validation import (
    CitationMatchAudit,
    CitationMatchStage,
    CitationValidationMode,
    audit_claim_citation,
    citation_matches,
)
from .cleanroom import build_cleanroom_review, load_cleanroom_adjudication
from .cleanroom_benchmark import (
    DEFAULT_CLEANROOM_ADJUDICATION,
    _load_cached_records,
    validate_cleanroom_adjudication,
)
from .evaluation import evaluate_record
from .provider import GenerationRecord

CITATION_AUDIT_VERSION = "citation-normalization-v1"
DEFAULT_MANUAL_AUDIT = PROJECT_ROOT / "experiments/samples/rag_behaviour_citation_manual_audit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/results/rag_baseline_behaviour_citation_audit.json"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ManualClassification(StrEnum):
    EVALUATOR_ARTIFACT = "evaluator_artifact"
    GENUINE_CITATION_ERROR = "genuine_citation_error"
    AMBIGUOUS = "ambiguous"


class FailureCategory(StrEnum):
    EVALUATOR_ARTIFACT = "evaluator_artifact"
    WRONG_CITATION_TARGET = "wrong_citation_target"
    UNSUPPORTED_PROPOSITION = "unsupported_proposition"
    QUOTE_NOT_FOUND = "quote_not_found"
    CORRECT_AUTHORITY_WRONG_LOCAL_PASSAGE = "correct_authority_wrong_local_passage"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OTHER = "other"


class ManualCitationAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    claim_number: int = Field(ge=1)
    evidence_id: str
    automatic_stage: CitationMatchStage
    classification: ManualClassification
    failure_category: FailureCategory
    claim_supported_by_cited_passage: bool | None
    citation_target_valid: bool
    authority_supplied_in_evidence: bool
    corrupted_representation: str | None
    canonical_representation: str | None
    reason: str = Field(min_length=1)


class ManualCitationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    audit_version: Literal["citation-normalization-v1"]
    source_run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")
    cleanroom_adjudication_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    review_date: date
    records: tuple[ManualCitationAuditRecord, ...] = Field(min_length=1)
    digest_algorithm: Literal["sha256-canonical-json-v1"]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> ManualCitationAudit:
        keys = [(record.package_id, record.claim_number) for record in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("manual citation audit records must be unique")
        if manual_audit_digest(self) != self.digest:
            raise ValueError("manual citation audit digest mismatch")
        return self


def manual_audit_digest(audit: ManualCitationAudit) -> str:
    return canonical_digest(audit.model_dump(mode="json", exclude={"digest"}))


def load_manual_audit(path: Path) -> ManualCitationAudit:
    return ManualCitationAudit.model_validate_json(path.read_text(encoding="utf-8"))


def _mean(values: list[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _relative_change(before: float | None, after: float | None) -> float | None:
    if before in {None, 0} or after is None:
        return None
    return (after - before) / before


def _claim_audits(record: GenerationRecord) -> list[CitationMatchAudit]:
    answer = record.result.answer
    if answer is None:
        return []
    return [
        audit_claim_citation(
            record.package,
            claim,
            recommended_case_id=answer.recommended_case_id,
        )
        for claim in answer.claims
    ]


def citation_metrics(records: list[GenerationRecord]) -> dict[str, Any]:
    strict_validity: list[bool] = []
    normalized_validity: list[bool] = []
    strict_correctness: list[float] = []
    normalized_correctness: list[float] = []
    completeness: list[float] = []
    strict_unsupported: list[float] = []
    normalized_unsupported: list[float] = []
    per_claim: list[dict[str, Any]] = []
    for record in records:
        answer = record.result.answer
        if answer is None or not answer.claims:
            continue
        outcome = evaluate_record(record)
        audits = _claim_audits(record)
        strict_valid = sum(
            citation_matches(audit, CitationValidationMode.STRICT) for audit in audits
        )
        normalized_valid = sum(
            citation_matches(audit, CitationValidationMode.NORMALIZED) for audit in audits
        )
        total = len(audits)
        non_quote_issues = [
            issue for issue in outcome.validation.issues if issue.code != "quote_not_verbatim"
        ]
        strict_record_valid = outcome.validation.structurally_valid
        normalized_record_valid = normalized_valid == total and not non_quote_issues
        strict_record_correctness = strict_valid / total
        normalized_record_correctness = normalized_valid / total
        strict_record_unsupported = 1.0 if non_quote_issues else 1.0 - strict_record_correctness
        normalized_record_unsupported = (
            1.0 if non_quote_issues else 1.0 - normalized_record_correctness
        )
        strict_validity.append(strict_record_valid)
        normalized_validity.append(normalized_record_valid)
        strict_correctness.append(strict_record_correctness)
        normalized_correctness.append(normalized_record_correctness)
        completeness.append(outcome.validation.citation_completeness or 0.0)
        strict_unsupported.append(strict_record_unsupported)
        normalized_unsupported.append(normalized_record_unsupported)
        for claim_number, (claim, audit) in enumerate(
            zip(answer.claims, audits, strict=True), start=1
        ):
            per_claim.append(
                {
                    "package_id": record.package.package_id,
                    "claim_number": claim_number,
                    "evidence_id": claim.evidence_id,
                    **audit.model_dump(mode="json"),
                }
            )

    strict = {
        "answered_records": len(strict_validity),
        "fully_valid_answered_records": sum(strict_validity),
        "citation_validity": _mean(strict_validity),
        "citation_correctness": _mean(strict_correctness),
        "citation_completeness": _mean(completeness),
        "unsupported_claim_rate": _mean(strict_unsupported),
    }
    normalized = {
        "answered_records": len(normalized_validity),
        "fully_valid_answered_records": sum(normalized_validity),
        "citation_validity": _mean(normalized_validity),
        "citation_correctness": _mean(normalized_correctness),
        "citation_completeness": _mean(completeness),
        "unsupported_claim_rate": _mean(normalized_unsupported),
    }
    deltas = {
        key: {
            "absolute": normalized[key] - strict[key],
            "relative": _relative_change(float(strict[key]), float(normalized[key])),
        }
        for key in (
            "fully_valid_answered_records",
            "citation_validity",
            "citation_correctness",
            "citation_completeness",
            "unsupported_claim_rate",
        )
    }
    affected_claims = [
        claim
        for claim in per_claim
        if not claim["historical_strict_match"] and claim["normalized_match"]
    ]
    return {
        "strict": strict,
        "normalized": normalized,
        "deltas": deltas,
        "stage_counts": dict(
            sorted(Counter(claim["earliest_match_stage"] for claim in per_claim).items())
        ),
        "affected_claims": affected_claims,
        "affected_records": sorted({claim["package_id"] for claim in affected_claims}),
        "per_claim": per_claim,
    }


def validate_manual_audit(
    manual: ManualCitationAudit,
    *,
    run_signature: str,
    cleanroom_digest: str,
    per_claim: list[dict[str, Any]],
) -> None:
    if manual.source_run_signature != run_signature:
        raise ValueError("manual citation audit run signature changed")
    if manual.cleanroom_adjudication_digest != cleanroom_digest:
        raise ValueError("manual citation audit clean-room digest changed")
    automatic_failures = {
        (claim["package_id"], claim["claim_number"]): claim
        for claim in per_claim
        if not claim["historical_strict_match"]
    }
    manual_by_key = {(record.package_id, record.claim_number): record for record in manual.records}
    if manual_by_key.keys() != automatic_failures.keys():
        raise ValueError("manual citation audit must cover every strict failure exactly")
    for key, record in manual_by_key.items():
        automatic = automatic_failures[key]
        if (
            record.evidence_id != automatic["evidence_id"]
            or record.automatic_stage.value != automatic["earliest_match_stage"]
            or record.citation_target_valid != automatic["citation_target_valid"]
            or record.authority_supplied_in_evidence != automatic["authority_supplied_in_evidence"]
        ):
            raise ValueError(f"manual citation audit metadata changed: {key}")


def run_citation_audit(
    *,
    manual_audit_path: Path = DEFAULT_MANUAL_AUDIT,
) -> dict[str, Any]:
    config, pilot, packages, _ = preflight_frozen_behaviour_execution(
        rag_config_path=DEFAULT_CONFIG,
        global_manifest_path=DEFAULT_MANIFEST,
        behaviour_manifest_path=DEFAULT_BEHAVIOUR_PILOT,
        behaviour_packages_path=DEFAULT_BEHAVIOUR_PACKAGES,
        behaviour_adjudication_path=DEFAULT_BEHAVIOUR_ADJUDICATION,
        answer_adjudication_path=DEFAULT_PILOT_ADJUDICATION,
    )
    cleanroom = load_cleanroom_adjudication(DEFAULT_CLEANROOM_ADJUDICATION)
    review = build_cleanroom_review(packages)
    validate_cleanroom_adjudication(
        cleanroom,
        review=review,
        pilot=pilot,
        packages=packages,
    )
    cached = _load_cached_records(
        cache_dir=DEFAULT_CACHE,
        run_signature=pilot.run_signature,
        packages=packages,
        settings=config.settings,
    )
    metrics = citation_metrics(cached)
    manual = load_manual_audit(manual_audit_path)
    validate_manual_audit(
        manual,
        run_signature=pilot.run_signature,
        cleanroom_digest=cleanroom.digest,
        per_claim=metrics["per_claim"],
    )
    return {
        "schema_version": 1,
        "audit_version": CITATION_AUDIT_VERSION,
        "historical_run_signature": pilot.run_signature,
        "historical_evidence_digest": pilot.evidence_digest,
        "historical_generation_contract": pilot.generation_contract,
        "cleanroom_adjudication_digest": cleanroom.digest,
        "strict_mode_preserved": True,
        "model_visible_inputs_changed": False,
        "provider_calls_made": 0,
        "metrics": metrics,
        "manual_audit_digest": manual.digest,
        "manual_classifications": [record.model_dump(mode="json") for record in manual.records],
        "manual_classification_counts": dict(
            sorted(Counter(record.classification.value for record in manual.records).items())
        ),
        "remaining_failure_taxonomy": dict(
            sorted(
                Counter(
                    record.failure_category.value
                    for record in manual.records
                    if record.classification is not ManualClassification.EVALUATOR_ARTIFACT
                ).items()
            )
        ),
        "type_b_not_applied": [
            "modify model-visible evidence text",
            "expose citation-verification metadata in the prompt",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict and normalized citation matching")
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_MANUAL_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_citation_audit(manual_audit_path=args.manual_audit)
        write_json(args.output, result)
        print(f"wrote citation validation audit {args.output}; no provider constructed")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"citation validation audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
