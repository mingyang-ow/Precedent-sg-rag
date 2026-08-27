from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .evidence import EvidenceCondition, EvidencePackage, ExpectedAction
from .provider import GenerationRecord, ProviderCallStatus
from .schema import AnswerStatus, GroundedAnswer

CASE_ID_RE = re.compile(r"case:[0-9]+")
LEGAL_CITATION_RE = re.compile(
    r"\[[12][0-9]{3}\]\s+(?:SG(?:CA|CAI|HC|HCF|HCR)\s+[0-9]+|[0-9]+\s+SLR(?:\(R\))?\s+[0-9]+)",
    re.IGNORECASE,
)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str


class AnswerValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    structurally_valid: bool
    valid_citations: int = Field(ge=0)
    total_citations: int = Field(ge=0)
    citation_correctness: float | None = Field(default=None, ge=0, le=1)
    citation_completeness: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate_proxy: float | None = Field(default=None, ge=0, le=1)
    issues: tuple[ValidationIssue, ...]


class EvaluationStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    PROVIDER_API_FAILURE = "provider_api_failure"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"


class EvaluationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    query_id: str
    mode: str
    condition: EvidenceCondition
    stratum: str
    top_k: int | None
    warm_start: bool
    target_present: bool
    evidence_sufficient: bool | None
    expected_action: ExpectedAction
    evaluation_status: EvaluationStatus
    provider_succeeded: bool
    answered: bool
    abstained: bool
    validation: AnswerValidation
    precedent_correct: bool | None
    grounded_generation_correct: bool | None
    grounded_end_to_end_success: bool | None
    abstention_correct: bool | None
    inappropriate_answer: bool | None
    primary_failure_layer: str
    retrieval_generation_layer: str | None
    abstention_layer: str | None


def normalize_space(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_answer(package: EvidencePackage, answer: GroundedAnswer | None) -> AnswerValidation:
    if answer is None:
        return AnswerValidation(
            structurally_valid=False,
            valid_citations=0,
            total_citations=0,
            citation_correctness=None,
            citation_completeness=None,
            unsupported_claim_rate_proxy=None,
            issues=(ValidationIssue(code="missing_answer", message="no parsed answer"),),
        )

    issues: list[ValidationIssue] = []
    evidence_by_id = {item.evidence_id: item for item in package.evidence}
    supplied_case_ids = {item.case_id for item in package.evidence}
    valid_citations = 0

    if (
        answer.recommended_case_id is not None
        and answer.recommended_case_id not in supplied_case_ids
    ):
        issues.append(
            ValidationIssue(
                code="unsupplied_recommendation",
                message=f"recommended case {answer.recommended_case_id} was not supplied",
            )
        )

    for position, claim in enumerate(answer.claims):
        evidence = evidence_by_id.get(claim.evidence_id)
        if evidence is None:
            issues.append(
                ValidationIssue(
                    code="unknown_evidence_id",
                    message=f"claim {position + 1} cites unknown {claim.evidence_id}",
                )
            )
            continue
        quote = normalize_space(claim.supporting_quote)
        passage = normalize_space(evidence.passage)
        if quote not in passage:
            issues.append(
                ValidationIssue(
                    code="quote_not_verbatim",
                    message=f"claim {position + 1} quote is not in {claim.evidence_id}",
                )
            )
            continue
        valid_citations += 1

    rendered = " ".join(
        [answer.explanation]
        + [claim.statement for claim in answer.claims]
        + [claim.supporting_quote for claim in answer.claims]
    )
    unseen_case_ids = set(CASE_ID_RE.findall(rendered)) - supplied_case_ids
    if unseen_case_ids:
        issues.append(
            ValidationIssue(
                code="unseen_case_id",
                message=f"output mentions unsupplied case IDs: {sorted(unseen_case_ids)}",
            )
        )
    allowed_citations = {
        match.group(0).casefold()
        for item in package.evidence
        for text in (item.case_name, item.source_judgment)
        for match in LEGAL_CITATION_RE.finditer(text)
    }
    unseen_citations = {
        match.group(0)
        for match in LEGAL_CITATION_RE.finditer(rendered)
        if match.group(0).casefold() not in allowed_citations
    }
    if unseen_citations:
        issues.append(
            ValidationIssue(
                code="unseen_legal_citation",
                message=f"output mentions unsupplied legal citations: {sorted(unseen_citations)}",
            )
        )

    total = len(answer.claims)
    correctness = valid_citations / total if total else None
    completeness = 1.0 if answer.status is AnswerStatus.ANSWERED and total else None
    unsupported = 1.0 - correctness if correctness is not None else None
    if unseen_case_ids or unseen_citations:
        unsupported = 1.0
    return AnswerValidation(
        structurally_valid=not issues,
        valid_citations=valid_citations,
        total_citations=total,
        citation_correctness=correctness,
        citation_completeness=completeness,
        unsupported_claim_rate_proxy=unsupported,
        issues=tuple(issues),
    )


def evaluate_record(record: GenerationRecord) -> EvaluationOutcome:
    package = record.package
    answer = record.result.answer
    validation = validate_answer(package, answer)
    call_status = record.result.call_status
    provider_succeeded = call_status is ProviderCallStatus.SUCCEEDED
    answered = answer is not None and answer.status is AnswerStatus.ANSWERED
    abstained = answer is not None and answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    if call_status is ProviderCallStatus.PROVIDER_API_FAILURE:
        evaluation_status = EvaluationStatus.PROVIDER_API_FAILURE
    elif call_status is ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE:
        evaluation_status = EvaluationStatus.STRUCTURED_OUTPUT_FAILURE
    elif answered:
        evaluation_status = EvaluationStatus.ANSWERED
    else:
        evaluation_status = EvaluationStatus.ABSTAINED
    answer_expected = package.expected_action is ExpectedAction.ANSWER
    abstention_expected = package.expected_action is ExpectedAction.ABSTAIN
    behaviour_known = package.expected_action is not ExpectedAction.UNKNOWN_NEEDS_REVIEW
    precedent_correct = (
        bool(
            answered
            and answer is not None
            and answer.recommended_case_id in package.accepted_case_ids
        )
        if answer_expected
        else None
    )
    fully_cited = validation.citation_correctness == 1.0
    generation_correct = (
        bool(
            provider_succeeded
            and answered
            and precedent_correct
            and validation.structurally_valid
            and fully_cited
        )
        if answer_expected
        else None
    )
    end_to_end = (
        bool(package.target_present and generation_correct)
        if generation_correct is not None
        else None
    )
    abstention_correct = (
        bool(abstained and validation.structurally_valid) if abstention_expected else None
    )
    inappropriate_answer = bool(answered) if abstention_expected else None

    retrieval_generation_layer: str | None
    if not provider_succeeded or not answer_expected:
        retrieval_generation_layer = None
    elif package.target_present:
        retrieval_generation_layer = (
            "1_target_present_generation_correct"
            if generation_correct
            else "2_target_present_generation_incorrect"
        )
    elif answered:
        grounded_to_supplied = validation.structurally_valid and fully_cited
        retrieval_generation_layer = (
            "3_target_absent_generation_grounded_to_supplied_evidence"
            if grounded_to_supplied
            else "4_target_absent_generation_unsupported"
        )
    else:
        retrieval_generation_layer = None

    abstention_layer = None
    if abstention_expected and provider_succeeded:
        abstention_layer = (
            "5_insufficient_evidence_correct_abstention"
            if abstention_correct
            else "6_insufficient_evidence_inappropriate_answer"
        )
    if evaluation_status is EvaluationStatus.PROVIDER_API_FAILURE:
        primary = "0_provider_api_failure"
    elif evaluation_status is EvaluationStatus.STRUCTURED_OUTPUT_FAILURE:
        primary = "0_structured_output_failure"
    elif not behaviour_known:
        primary = "7_evidence_sufficiency_unknown_needs_review"
    else:
        primary = (
            abstention_layer
            or retrieval_generation_layer
            or (
                "2_target_present_generation_incorrect"
                if package.target_present
                else "4_target_absent_generation_unsupported"
            )
        )
    return EvaluationOutcome(
        package_id=package.package_id,
        query_id=package.query_id,
        mode=package.query_mode,
        condition=package.condition,
        stratum=package.stratum,
        top_k=package.top_k,
        warm_start=package.warm_start,
        target_present=package.target_present,
        evidence_sufficient=package.evidence_sufficient,
        expected_action=package.expected_action,
        evaluation_status=evaluation_status,
        provider_succeeded=provider_succeeded,
        answered=answered,
        abstained=abstained,
        validation=validation,
        precedent_correct=precedent_correct,
        grounded_generation_correct=generation_correct,
        grounded_end_to_end_success=end_to_end,
        abstention_correct=abstention_correct,
        inappropriate_answer=inappropriate_answer,
        primary_failure_layer=primary,
        retrieval_generation_layer=retrieval_generation_layer,
        abstention_layer=abstention_layer,
    )


def _mean(values: Iterable[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def behaviour_metrics(outcomes: list[EvaluationOutcome]) -> dict[str, object]:
    """Score the answer/abstain decision boundary without folding in provider failures."""

    known = [
        outcome
        for outcome in outcomes
        if outcome.provider_succeeded
        and outcome.expected_action is not ExpectedAction.UNKNOWN_NEEDS_REVIEW
    ]
    true_positive = sum(
        outcome.expected_action is ExpectedAction.ANSWER and outcome.answered for outcome in known
    )
    false_negative = sum(
        outcome.expected_action is ExpectedAction.ANSWER and outcome.abstained for outcome in known
    )
    false_positive = sum(
        outcome.expected_action is ExpectedAction.ABSTAIN and outcome.answered for outcome in known
    )
    true_negative = sum(
        outcome.expected_action is ExpectedAction.ABSTAIN and outcome.abstained for outcome in known
    )
    answer_total = true_positive + false_negative
    abstain_total = true_negative + false_positive
    answer_recall = true_positive / answer_total if answer_total else None
    abstention_recall = true_negative / abstain_total if abstain_total else None
    return {
        "confusion_matrix": {
            "true_positive_answer": true_positive,
            "false_negative_abstention": false_negative,
            "false_positive_answer": false_positive,
            "true_negative_abstention": true_negative,
        },
        "evaluable_records": len(known),
        "excluded_provider_api_failures": sum(
            outcome.evaluation_status is EvaluationStatus.PROVIDER_API_FAILURE
            for outcome in outcomes
        ),
        "excluded_structured_output_failures": sum(
            outcome.evaluation_status is EvaluationStatus.STRUCTURED_OUTPUT_FAILURE
            for outcome in outcomes
        ),
        "excluded_unknown_ground_truth": sum(
            outcome.provider_succeeded
            and outcome.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
            for outcome in outcomes
        ),
        "answer_recall": answer_recall,
        "abstention_recall": abstention_recall,
        "false_answer_rate": false_positive / abstain_total if abstain_total else None,
        "false_abstention_rate": false_negative / answer_total if answer_total else None,
        "balanced_accuracy": (
            (answer_recall + abstention_recall) / 2
            if answer_recall is not None and abstention_recall is not None
            else None
        ),
        "unsupported_claim_rate": _mean(
            outcome.validation.unsupported_claim_rate_proxy for outcome in known
        ),
        "citation_validity": _mean(
            outcome.validation.structurally_valid for outcome in known if outcome.answered
        ),
        "citation_correctness": _mean(outcome.validation.citation_correctness for outcome in known),
        "citation_completeness": _mean(
            outcome.validation.citation_completeness for outcome in known
        ),
    }


def summarize_records(records: list[GenerationRecord]) -> dict[str, object]:
    outcomes = [evaluate_record(record) for record in records]
    evaluable = [outcome for outcome in outcomes if outcome.provider_succeeded]
    expected_answers = [
        outcome for outcome in evaluable if outcome.expected_action is ExpectedAction.ANSWER
    ]
    expected_insufficient = [
        outcome for outcome in evaluable if outcome.expected_action is ExpectedAction.ABSTAIN
    ]
    behaviour_known = [
        outcome
        for outcome in evaluable
        if outcome.expected_action is not ExpectedAction.UNKNOWN_NEEDS_REVIEW
    ]
    predicted_abstentions = [outcome for outcome in behaviour_known if outcome.abstained]
    correct_abstentions = [
        outcome for outcome in expected_insufficient if outcome.abstention_correct
    ]
    retrieved = [
        outcome for outcome in expected_answers if outcome.condition is EvidenceCondition.RETRIEVED
    ]
    latencies = [record.result.latency_ms for record in records]
    usage = [record.result.usage for record in records if record.result.usage is not None]
    costs = [
        record.result.estimated_cost_usd
        for record in records
        if record.result.estimated_cost_usd is not None
    ]
    return {
        "records": len(records),
        "evaluable_records": len(evaluable),
        "expected_actions": dict(
            sorted(Counter(outcome.expected_action.value for outcome in outcomes).items())
        ),
        "manual_sufficiency_review_records": sum(
            outcome.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW for outcome in outcomes
        ),
        "target_present_records": sum(outcome.target_present for outcome in outcomes),
        "evaluation_statuses": dict(
            sorted(Counter(outcome.evaluation_status.value for outcome in outcomes).items())
        ),
        "provider_api_failures": sum(
            outcome.evaluation_status is EvaluationStatus.PROVIDER_API_FAILURE
            for outcome in outcomes
        ),
        "structured_output_failures": sum(
            outcome.evaluation_status is EvaluationStatus.STRUCTURED_OUTPUT_FAILURE
            for outcome in outcomes
        ),
        "provider_success_rate": _mean(outcome.provider_succeeded for outcome in outcomes),
        "answer_rate": _mean(outcome.answered for outcome in evaluable),
        "faithfulness_proxy": _mean(
            outcome.validation.citation_correctness for outcome in outcomes
        ),
        "citation_correctness": _mean(
            outcome.validation.citation_correctness for outcome in outcomes
        ),
        "citation_completeness": _mean(
            outcome.validation.citation_completeness for outcome in outcomes
        ),
        "unsupported_claim_rate_proxy": _mean(
            outcome.validation.unsupported_claim_rate_proxy for outcome in outcomes
        ),
        "precedent_correctness": _mean(outcome.precedent_correct for outcome in expected_answers),
        "grounded_generation_success": _mean(
            outcome.grounded_generation_correct for outcome in expected_answers
        ),
        "grounded_end_to_end_success": _mean(
            outcome.grounded_end_to_end_success for outcome in retrieved
        ),
        "abstention_precision": (
            len(correct_abstentions) / len(predicted_abstentions) if predicted_abstentions else None
        ),
        "abstention_recall": (
            len(correct_abstentions) / len(expected_insufficient) if expected_insufficient else None
        ),
        "inappropriate_answer_rate": _mean(
            outcome.inappropriate_answer for outcome in expected_insufficient
        ),
        "primary_failure_layers": dict(
            sorted(Counter(outcome.primary_failure_layer for outcome in outcomes).items())
        ),
        "retrieval_generation_layers": dict(
            sorted(
                Counter(
                    outcome.retrieval_generation_layer
                    for outcome in outcomes
                    if outcome.retrieval_generation_layer is not None
                ).items()
            )
        ),
        "abstention_layers": dict(
            sorted(
                Counter(
                    outcome.abstention_layer
                    for outcome in outcomes
                    if outcome.abstention_layer is not None
                ).items()
            )
        ),
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "tokens": {
            "input": sum(item.input_tokens for item in usage),
            "cached_input": sum(item.cached_input_tokens for item in usage),
            "output": sum(item.output_tokens for item in usage),
            "reasoning": sum(item.reasoning_tokens for item in usage),
            "total": sum(item.total_tokens for item in usage),
        },
        "estimated_cost_usd": sum(costs),
        "behaviour": behaviour_metrics(outcomes),
    }


def grouped_summaries(records: list[GenerationRecord]) -> dict[str, object]:
    def groups(key: str) -> dict[str, dict[str, object]]:
        values: dict[str, list[GenerationRecord]] = {}
        for record in records:
            package = record.package
            if key == "mode":
                label = package.query_mode
            elif key == "condition":
                label = package.condition.value
            elif key == "top_k":
                label = "oracle" if package.top_k is None else str(package.top_k)
            elif key == "warm_cold":
                label = "warm" if package.warm_start else "cold"
            else:
                raise ValueError(f"unknown grouping: {key}")
            values.setdefault(label, []).append(record)
        return {label: summarize_records(items) for label, items in sorted(values.items())}

    return {
        "overall": summarize_records(records),
        "by_mode": groups("mode"),
        "by_condition": groups("condition"),
        "by_top_k": groups("top_k"),
        "by_warm_cold": groups("warm_cold"),
    }
