from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .behaviour_pilot import (
    BehaviourAdjudication,
    BehaviourPilotManifest,
    load_behaviour_adjudication,
)
from .benchmark import (
    DEFAULT_BEHAVIOUR_ADJUDICATION,
    DEFAULT_BEHAVIOUR_PACKAGES,
    DEFAULT_BEHAVIOUR_PILOT,
    DEFAULT_CACHE,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_PILOT_ADJUDICATION,
    PROJECT_ROOT,
    assert_cached_record_matches_execution,
    preflight_frozen_behaviour_execution,
    write_json,
)
from .cleanroom import (
    CLEANROOM_ADJUDICATION_VERSION,
    CleanroomAdjudication,
    CleanroomLabel,
    CleanroomReviewArtifact,
    build_cleanroom_review,
    cleanroom_review_digest,
    load_cleanroom_adjudication,
    render_cleanroom_review,
)
from .evaluation import EvaluationOutcome, evaluate_record, grouped_summaries, normalize_space
from .evidence import EvidencePackage, EvidenceSufficiencyBasis, ExpectedAction
from .provider import GenerationRecord, GenerationSettings, cache_path, load_record

DEFAULT_CLEANROOM_REVIEW = PROJECT_ROOT / ".local/cleanroom/behaviour-pilot-review.md"
DEFAULT_CLEANROOM_ADJUDICATION = (
    PROJECT_ROOT / "experiments/samples/rag_behaviour_cleanroom_adjudication.json"
)
DEFAULT_CLEANROOM_OUTPUT = (
    PROJECT_ROOT / "experiments/results/rag_baseline_behaviour_cleanroom.json"
)


def validate_cleanroom_adjudication(
    adjudication: CleanroomAdjudication,
    *,
    review: CleanroomReviewArtifact,
    pilot: BehaviourPilotManifest,
    packages: tuple[EvidencePackage, ...],
) -> None:
    if adjudication.adjudication_version != CLEANROOM_ADJUDICATION_VERSION:
        raise ValueError("clean-room adjudication version changed")
    if adjudication.source_run_signature != pilot.run_signature:
        raise ValueError("clean-room source run signature changed")
    if adjudication.source_evidence_digest != pilot.evidence_digest:
        raise ValueError("clean-room source evidence digest changed")
    if adjudication.review_artifact_digest != cleanroom_review_digest(review):
        raise ValueError("clean-room review artifact digest changed")
    package_ids = tuple(package.package_id for package in packages)
    record_ids = tuple(record.package_id for record in adjudication.records)
    if record_ids != package_ids:
        raise ValueError("clean-room package IDs or ordering changed")
    review_ids = tuple(record.package_id for record in review.records)
    if review_ids != package_ids:
        raise ValueError("clean-room review package IDs or ordering changed")
    for package, record in zip(packages, adjudication.records, strict=True):
        evidence_ids = {item.evidence_id for item in package.evidence}
        if not set(record.supporting_evidence_ids).issubset(evidence_ids):
            raise ValueError(f"clean-room review cites unknown evidence: {package.package_id}")


def package_with_cleanroom_label(
    package: EvidencePackage, label: CleanroomLabel
) -> EvidencePackage:
    payload = package.model_dump(mode="python")
    if label is CleanroomLabel.ANSWER:
        payload.update(
            evidence_sufficient=True,
            expected_action=ExpectedAction.ANSWER,
            sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT,
        )
    elif label is CleanroomLabel.ABSTAIN:
        payload.update(
            evidence_sufficient=False,
            expected_action=ExpectedAction.ABSTAIN,
            sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT,
        )
    else:
        payload.update(
            evidence_sufficient=None,
            expected_action=ExpectedAction.UNKNOWN_NEEDS_REVIEW,
            sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED,
        )
    return EvidencePackage.model_validate(payload)


def _load_cached_records(
    *,
    cache_dir: Path,
    run_signature: str,
    packages: tuple[EvidencePackage, ...],
    settings: GenerationSettings,
) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for package in packages:
        path = cache_path(cache_dir, run_signature, package.package_id)
        record = load_record(path, run_signature=run_signature, package_id=package.package_id)
        if record is None:
            raise ValueError(f"clean-room evaluation cache is missing: {package.package_id}")
        assert_cached_record_matches_execution(
            record,
            package=package,
            settings=settings,
            run_signature=run_signature,
        )
        records.append(record)
    return records


def _mean(values: list[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def output_quality(outcomes: list[EvaluationOutcome]) -> dict[str, Any]:
    answered = [outcome for outcome in outcomes if outcome.answered]
    hallucinated_codes = {"unsupplied_recommendation", "unseen_case_id", "unseen_legal_citation"}
    hallucinated = [
        outcome
        for outcome in answered
        if any(issue.code in hallucinated_codes for issue in outcome.validation.issues)
    ]
    return {
        "records": len(outcomes),
        "provider_successes": sum(outcome.provider_succeeded for outcome in outcomes),
        "provider_api_failures": sum(
            outcome.evaluation_status.value == "provider_api_failure" for outcome in outcomes
        ),
        "structured_output_failures": sum(
            outcome.evaluation_status.value == "structured_output_failure" for outcome in outcomes
        ),
        "answered_records": len(answered),
        "citation_validity": _mean([outcome.validation.structurally_valid for outcome in answered]),
        "citation_correctness": _mean(
            [outcome.validation.citation_correctness for outcome in answered]
        ),
        "citation_completeness": _mean(
            [outcome.validation.citation_completeness for outcome in answered]
        ),
        "unsupported_claim_rate": _mean(
            [outcome.validation.unsupported_claim_rate_proxy for outcome in answered]
        ),
        "hallucinated_authority_records": len(hallucinated),
        "hallucinated_authority_rate": len(hallucinated) / len(answered) if answered else None,
        "hallucinated_authority_package_ids": [outcome.package_id for outcome in hallucinated],
    }


def _mojibake_repair(text: str) -> str | None:
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return None
    return repaired if repaired != text else None


def normalization_audit(records: list[GenerationRecord]) -> dict[str, Any]:
    evaluator_only: list[dict[str, Any]] = []
    possible_mojibake: list[dict[str, Any]] = []
    total_claims = 0
    currently_valid_claims = 0
    candidate_valid_claims = 0
    answered_records = 0
    currently_valid_records = 0
    candidate_valid_records = 0
    for record in records:
        answer = record.result.answer
        if answer is None or not answer.claims:
            continue
        answered_records += 1
        evidence_by_id = {item.evidence_id: item for item in record.package.evidence}
        current_record_valid = True
        candidate_record_valid = True
        for position, claim in enumerate(answer.claims, start=1):
            total_claims += 1
            evidence = evidence_by_id.get(claim.evidence_id)
            if evidence is None:
                current_record_valid = False
                candidate_record_valid = False
                continue
            exact = claim.supporting_quote in evidence.passage
            normalized = normalize_space(claim.supporting_quote) in normalize_space(
                evidence.passage
            )
            if normalized:
                currently_valid_claims += 1
                candidate_valid_claims += 1
            else:
                current_record_valid = False
            if not exact and normalized:
                evaluator_only.append(
                    {
                        "package_id": record.package.package_id,
                        "claim": position,
                        "evidence_id": claim.evidence_id,
                        "classification": "type_a_evaluator_only",
                        "reason": "NFKC/whitespace normalization matches unchanged visible text",
                    }
                )
            if normalized:
                continue
            repaired_passage = _mojibake_repair(evidence.passage)
            repaired_quote = _mojibake_repair(claim.supporting_quote)
            encoding_equivalent = (
                repaired_passage is not None
                and normalize_space(claim.supporting_quote) in normalize_space(repaired_passage)
            ) or (
                repaired_quote is not None
                and normalize_space(repaired_quote) in normalize_space(evidence.passage)
            )
            if encoding_equivalent:
                candidate_valid_claims += 1
                possible_mojibake.append(
                    {
                        "package_id": record.package.package_id,
                        "claim": position,
                        "evidence_id": claim.evidence_id,
                        "classification": "type_a_candidate_not_applied",
                        "reason": "encoding repair would change evaluator matching only",
                    }
                )
            else:
                candidate_record_valid = False
        currently_valid_records += current_record_valid
        candidate_valid_records += candidate_record_valid
    return {
        "type_a_applied": evaluator_only,
        "type_a_candidates_not_applied": possible_mojibake,
        "type_b_not_applied": [
            {
                "change": "normalize query, evidence, case/source identifiers, or formatting",
                "consequence": ("new evidence digest, run signature, and frozen pilot required"),
            },
            {
                "change": "expose citation-relationship diagnostics to the model",
                "consequence": ("new evidence digest, run signature, and frozen pilot required"),
            },
        ],
        "strict_contract_matching": {
            "valid_claims": currently_valid_claims,
            "total_claims": total_claims,
            "valid_answered_records": currently_valid_records,
            "answered_records": answered_records,
        },
        "encoding_equivalence_candidate_not_applied": {
            "valid_claims": candidate_valid_claims,
            "total_claims": total_claims,
            "valid_answered_records": candidate_valid_records,
            "answered_records": answered_records,
            "reason": (
                "reported separately because rag-v2 requires verbatim quotes; primary metrics "
                "remain strict"
            ),
        },
        "model_visible_evidence_changed": False,
    }


def recompute_cleanroom_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    manifest_path: Path = DEFAULT_MANIFEST,
    behaviour_manifest_path: Path = DEFAULT_BEHAVIOUR_PILOT,
    behaviour_packages_path: Path = DEFAULT_BEHAVIOUR_PACKAGES,
    behaviour_adjudication_path: Path = DEFAULT_BEHAVIOUR_ADJUDICATION,
    answer_adjudication_path: Path = DEFAULT_PILOT_ADJUDICATION,
    cleanroom_adjudication_path: Path = DEFAULT_CLEANROOM_ADJUDICATION,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    config, pilot, packages, _ = preflight_frozen_behaviour_execution(
        rag_config_path=config_path,
        global_manifest_path=manifest_path,
        behaviour_manifest_path=behaviour_manifest_path,
        behaviour_packages_path=behaviour_packages_path,
        behaviour_adjudication_path=behaviour_adjudication_path,
        answer_adjudication_path=answer_adjudication_path,
    )
    review = build_cleanroom_review(packages)
    cleanroom = load_cleanroom_adjudication(cleanroom_adjudication_path)
    validate_cleanroom_adjudication(cleanroom, review=review, pilot=pilot, packages=packages)

    # The cache boundary is intentionally below the fully frozen adjudication validation above.
    cached = _load_cached_records(
        cache_dir=cache_dir,
        run_signature=pilot.run_signature,
        packages=packages,
        settings=config.settings,
    )
    cleanroom_by_id = {record.package_id: record for record in cleanroom.records}
    labeled_records = [
        record.model_copy(
            update={
                "package": package_with_cleanroom_label(
                    record.package, cleanroom_by_id[record.package.package_id].label
                )
            }
        )
        for record in cached
    ]
    binary_records = [
        record
        for record in labeled_records
        if record.package.expected_action is not ExpectedAction.UNKNOWN_NEEDS_REVIEW
    ]
    all_outcomes = [evaluate_record(record) for record in labeled_records]
    uncertain = [
        {
            "package_id": decision.package_id,
            "label": decision.label.value,
            "rationale": decision.rationale,
            "model_status": next(
                outcome.evaluation_status.value
                for outcome in all_outcomes
                if outcome.package_id == decision.package_id
            ),
        }
        for decision in cleanroom.records
        if decision.label in {CleanroomLabel.BORDERLINE, CleanroomLabel.CANNOT_DETERMINE}
    ]

    original: BehaviourAdjudication = load_behaviour_adjudication(behaviour_adjudication_path)
    original_by_id = {
        record.package_id: record
        for record in original.reviewed_candidates + original.oracle_reviews
    }
    changed_labels = [
        {
            "package_id": decision.package_id,
            "original": original_by_id[decision.package_id].expected_action.value,
            "cleanroom": decision.label.value,
            "cleanroom_rationale": decision.rationale,
        }
        for decision in cleanroom.records
        if decision.label.value != original_by_id[decision.package_id].expected_action.value
    ]
    label_counts = dict(sorted(Counter(record.label.value for record in cleanroom.records).items()))
    return {
        "schema_version": 1,
        "evaluation_version": CLEANROOM_ADJUDICATION_VERSION,
        "historical_run_signature": pilot.run_signature,
        "historical_evidence_digest": pilot.evidence_digest,
        "cleanroom_adjudication_digest": cleanroom.digest,
        "labels": label_counts,
        "changed_labels": changed_labels,
        "binary_records": len(binary_records),
        "excluded_uncertain_records": len(uncertain),
        "aggregates": grouped_summaries(binary_records),
        "output_quality_all_12": output_quality(all_outcomes),
        "uncertain_records": uncertain,
        "difficulty": {
            "clear_binary": grouped_summaries(binary_records)["overall"],
            "uncertain": uncertain,
        },
        "normalization_audit": normalization_audit(cached),
        "provider_calls_made_by_recomputation": 0,
    }


def export_cleanroom_review(
    *,
    output_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    manifest_path: Path = DEFAULT_MANIFEST,
    behaviour_manifest_path: Path = DEFAULT_BEHAVIOUR_PILOT,
    behaviour_packages_path: Path = DEFAULT_BEHAVIOUR_PACKAGES,
    behaviour_adjudication_path: Path = DEFAULT_BEHAVIOUR_ADJUDICATION,
    answer_adjudication_path: Path = DEFAULT_PILOT_ADJUDICATION,
) -> CleanroomReviewArtifact:
    _, _, packages, _ = preflight_frozen_behaviour_execution(
        rag_config_path=config_path,
        global_manifest_path=manifest_path,
        behaviour_manifest_path=behaviour_manifest_path,
        behaviour_packages_path=behaviour_packages_path,
        behaviour_adjudication_path=behaviour_adjudication_path,
        answer_adjudication_path=answer_adjudication_path,
    )
    review = build_cleanroom_review(packages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_cleanroom_review(review), encoding="utf-8")
    return review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind re-adjudication of a frozen RAG pilot")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export-review", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    parser.add_argument("--review-output", type=Path, default=DEFAULT_CLEANROOM_REVIEW)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_CLEANROOM_ADJUDICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEANROOM_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.export_review:
            review = export_cleanroom_review(output_path=args.review_output)
            print(
                f"wrote sanitized clean-room review {args.review_output}; "
                f"records={len(review.records)}; no cached outputs loaded"
            )
            return 0
        result = recompute_cleanroom_evaluation(cleanroom_adjudication_path=args.adjudication)
        write_json(args.output, result)
        print(f"wrote clean-room evaluation {args.output}; no provider constructed")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"clean-room evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
