from __future__ import annotations

from pathlib import Path

import pytest

from sg_legal_rag.generation import cleanroom_benchmark as cleanroom_module
from sg_legal_rag.generation.behaviour_pilot import (
    load_behaviour_pilot,
    load_frozen_behaviour_packages,
)
from sg_legal_rag.generation.benchmark import (
    DEFAULT_BEHAVIOUR_PACKAGES,
    DEFAULT_BEHAVIOUR_PILOT,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    load_config,
    load_json,
    package_evidence_lock,
)
from sg_legal_rag.generation.cleanroom import (
    CleanroomLabel,
    build_cleanroom_review,
    cleanroom_adjudication_digest,
    load_cleanroom_adjudication,
    render_cleanroom_review,
)
from sg_legal_rag.generation.cleanroom_benchmark import (
    DEFAULT_CLEANROOM_ADJUDICATION,
    normalization_audit,
    package_with_cleanroom_label,
)
from sg_legal_rag.generation.evaluation import behaviour_metrics, evaluate_record
from sg_legal_rag.generation.provider import (
    GenerationRecord,
    ProviderCallStatus,
    ProviderResult,
)
from sg_legal_rag.generation.schema import AnswerStatus, GroundedAnswer, GroundedClaim

FORBIDDEN_REVIEW_TEXT = {
    "citation_relationship_verified",
    "target_present",
    "expected_action",
    "evidence_sufficient",
    "review_rationale",
    "provider_status",
    "response_id",
    "raw_output",
    "retrieval_rank",
    "retrieval_score",
    "gold_row_id",
    "accepted_case_ids",
}


def frozen_packages():
    return load_frozen_behaviour_packages(DEFAULT_BEHAVIOUR_PACKAGES).packages


def test_cleanroom_review_export_contains_only_model_visible_material() -> None:
    rendered = render_cleanroom_review(build_cleanroom_review(frozen_packages()))

    assert all(field not in rendered for field in FORBIDDEN_REVIEW_TEXT)
    assert "previous" not in rendered.casefold()
    assert "model response" not in rendered.casefold()
    assert rendered.count("## Record ") == 12


def test_cleanroom_review_reproduces_frozen_rendered_input_signatures() -> None:
    packages = frozen_packages()
    review = build_cleanroom_review(packages)
    pilot = load_behaviour_pilot(DEFAULT_BEHAVIOUR_PILOT)
    manifest = load_json(DEFAULT_MANIFEST)
    locks = {lock["package_id"]: lock for lock in manifest["evidence_freeze"]["packages"]}

    assert tuple(record.package_id for record in review.records) == pilot.selected_package_ids
    assert all(
        record.rendered_input_signature == locks[record.package_id]["input_signature"]
        for record in review.records
    )
    assert all(package_evidence_lock(package) == locks[package.package_id] for package in packages)


def test_cleanroom_adjudication_digest_is_deterministic_and_frozen() -> None:
    first = load_cleanroom_adjudication(DEFAULT_CLEANROOM_ADJUDICATION)
    second = load_cleanroom_adjudication(DEFAULT_CLEANROOM_ADJUDICATION)

    assert first == second
    assert cleanroom_adjudication_digest(first) == first.digest
    assert first.digest == "f3915f8202a56f687cc85655290532d36da6603c7c636e57b8500a90845eb6db"


def _record(package, *, answered: bool) -> GenerationRecord:
    settings = load_config(DEFAULT_CONFIG).settings
    item = package.evidence[0]
    answer = (
        GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            recommended_case_id=item.case_id,
            explanation="The supplied authority is relevant.",
            claims=[
                GroundedClaim(
                    statement="The authority states a relevant proposition.",
                    evidence_id=item.evidence_id,
                    supporting_quote=item.passage[:20],
                )
            ],
        )
        if answered
        else GroundedAnswer(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            recommended_case_id=None,
            explanation="The supplied evidence does not support a precedent recommendation.",
            claims=[],
        )
    )
    return GenerationRecord(
        run_signature="test-cleanroom",
        package=package,
        prompt_version=settings.prompt_version,
        system_instructions="test",
        user_input="test",
        settings=settings,
        result=ProviderResult(
            requested_model=settings.model,
            returned_model=settings.model,
            response_id="test",
            generated_at="2026-08-27T00:00:00+00:00",
            latency_ms=0,
            usage=None,
            estimated_cost_usd=0,
            raw_output=answer.model_dump_json(),
            answer=answer,
            error=None,
            provider_status=ProviderCallStatus.SUCCEEDED,
        ),
    )


def test_uncertain_cleanroom_labels_are_excluded_from_binary_denominators() -> None:
    base = frozen_packages()[0]
    labels_and_outputs = (
        (CleanroomLabel.ANSWER, True),
        (CleanroomLabel.ABSTAIN, False),
        (CleanroomLabel.BORDERLINE, True),
        (CleanroomLabel.CANNOT_DETERMINE, False),
    )
    outcomes = [
        evaluate_record(_record(package_with_cleanroom_label(base, label), answered=answered))
        for label, answered in labels_and_outputs
    ]

    metrics = behaviour_metrics(outcomes)

    assert metrics["evaluable_records"] == 2
    assert metrics["excluded_unknown_ground_truth"] == 2
    assert metrics["confusion_matrix"] == {
        "true_positive_answer": 1,
        "false_negative_abstention": 0,
        "false_positive_answer": 0,
        "true_negative_abstention": 1,
    }


def test_cached_outputs_are_below_the_adjudication_freeze_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_load = cleanroom_module.load_cleanroom_adjudication
    original_validate = cleanroom_module.validate_cleanroom_adjudication

    def load(path: Path):
        events.append("load_adjudication")
        return original_load(path)

    def validate(*args, **kwargs):
        events.append("validate_adjudication")
        return original_validate(*args, **kwargs)

    def cache(**kwargs):
        events.append("load_cache")
        raise RuntimeError("stop after proving boundary order")

    monkeypatch.setattr(cleanroom_module, "load_cleanroom_adjudication", load)
    monkeypatch.setattr(cleanroom_module, "validate_cleanroom_adjudication", validate)
    monkeypatch.setattr(cleanroom_module, "_load_cached_records", cache)

    with pytest.raises(RuntimeError, match="proving boundary"):
        cleanroom_module.recompute_cleanroom_evaluation()

    assert events == ["load_adjudication", "validate_adjudication", "load_cache"]


def test_cleanroom_evaluator_has_no_provider_constructor() -> None:
    assert "OpenAIResponsesGenerator" not in cleanroom_module.__dict__


def test_evaluator_only_normalization_does_not_change_model_visible_signatures() -> None:
    package = frozen_packages()[0]
    before = package_evidence_lock(package)
    normalization_audit([_record(package, answered=True)])

    assert package_evidence_lock(package) == before
