from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sg_legal_rag.generation import benchmark as benchmark_module
from sg_legal_rag.generation.benchmark import (
    RAGConfig,
    evidence_freeze,
    manual_review_template,
)
from sg_legal_rag.generation.evaluation import evaluate_record
from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidenceItem,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)
from sg_legal_rag.generation.provider import (
    SYSTEM_INSTRUCTIONS,
    GenerationRecord,
    GenerationSettings,
    ProviderCallStatus,
    ProviderResult,
)
from sg_legal_rag.generation.schema import AnswerStatus, GroundedAnswer

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "rag_prompt_behaviour.json"
FIXTURE_PAYLOAD = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
FIXTURES = {item["id"]: item for item in FIXTURE_PAYLOAD["fixtures"]}


def settings() -> GenerationSettings:
    return GenerationSettings(
        model="gpt-5.6-luna",
        reasoning_effort="none",
        verbosity="low",
        max_output_tokens=600,
        prompt_version="rag-v2",
        input_usd_per_million=0.2,
        cached_input_usd_per_million=0.02,
        output_usd_per_million=1.2,
    )


def package_for(fixture: dict[str, object]) -> EvidencePackage:
    action = ExpectedAction(str(fixture["expected_action"]))
    sufficient = action is ExpectedAction.ANSWER
    condition = EvidenceCondition(str(fixture["condition"]))
    origin = (
        EvidenceOrigin.GOLD_QUERY_ROW
        if condition is EvidenceCondition.ORACLE_GOLD
        else EvidenceOrigin.HISTORICAL_RETRIEVAL
    )
    passage = str(fixture["passage"])
    case_id = str(fixture["case_id"])
    evidence = EvidenceItem(
        evidence_id="E1",
        case_id=case_id,
        case_name=str(fixture["case_name"]),
        source_judgment="Synthetic prompt-behaviour fixture",
        source_url=f"https://fixture.invalid/{fixture['id']}",
        source_year=2024 if origin is EvidenceOrigin.GOLD_QUERY_ROW else 2023,
        passage=passage,
        passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        retrieval_rank=None if origin is EvidenceOrigin.GOLD_QUERY_ROW else 1,
        retrieval_score=None if origin is EvidenceOrigin.GOLD_QUERY_ROW else 1.0,
        origin=origin,
        gold_row_id=(
            f"synthetic-{fixture['id']}" if origin is EvidenceOrigin.GOLD_QUERY_ROW else None
        ),
        citation_relationship_verified=True,
    )
    return EvidencePackage(
        package_id=f"fixture-{fixture['id']}",
        query_id=f"query-{fixture['id']}",
        query_mode="facts_only",
        query_text=str(fixture["query"]),
        stratum="prompt_behaviour_fixture",
        condition=condition,
        top_k=None if condition is EvidenceCondition.ORACLE_GOLD else 1,
        evidence=(evidence,),
        accepted_case_ids=(case_id,),
        warm_start=True,
        target_present=True,
        evidence_sufficient=sufficient,
        expected_action=action,
        sufficiency_basis=(
            EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT
            if sufficient
            else EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT
        ),
    )


def record_for(fixture: dict[str, object]) -> GenerationRecord:
    answer = GroundedAnswer.model_validate(fixture["answer"])
    generation_settings = settings()
    return GenerationRecord(
        run_signature="fixture-signature",
        package=package_for(fixture),
        prompt_version=generation_settings.prompt_version,
        system_instructions=SYSTEM_INSTRUCTIONS,
        user_input="synthetic offline fixture",
        settings=generation_settings,
        result=ProviderResult(
            requested_model=generation_settings.model,
            returned_model=generation_settings.model,
            response_id="fixture-response",
            generated_at="2026-08-26T00:00:00+00:00",
            latency_ms=0,
            usage=None,
            estimated_cost_usd=None,
            raw_output=answer.model_dump_json(),
            answer=answer,
            error=None,
            provider_status=ProviderCallStatus.SUCCEEDED,
        ),
    )


def test_prompt_states_the_relevance_not_merits_answerability_threshold() -> None:
    prompt = " ".join(SYSTEM_INSTRUCTIONS.split())

    assert "relevant authority for the legal principle, rule, or test" in prompt
    assert "Do not decide whether the client's facts ultimately satisfy" in prompt
    assert "even if factual application remains unresolved" in prompt
    assert "Case identity alone is insufficient" in prompt
    assert "Abstain only when" in prompt


@pytest.mark.parametrize("fixture", FIXTURES.values(), ids=FIXTURES)
def test_offline_prompt_behaviour_fixture_has_the_expected_action(
    fixture: dict[str, object],
) -> None:
    outcome = evaluate_record(record_for(fixture))

    if fixture["expected_action"] == "answer":
        assert outcome.evaluation_status.value == "answered"
        assert outcome.precedent_correct
        assert outcome.grounded_generation_correct
        assert outcome.validation.structurally_valid
        assert outcome.validation.unsupported_claim_rate_proxy == 0
    else:
        assert outcome.evaluation_status.value == "abstained"
        assert outcome.abstention_correct
        assert outcome.primary_failure_layer == "5_insufficient_evidence_correct_abstention"


def test_relevant_test_and_unresolved_facts_both_expect_answers() -> None:
    assert FIXTURES["relevant_precedent_direct_test"]["expected_action"] == "answer"
    assert FIXTURES["relevant_test_unresolved_facts"]["expected_action"] == "answer"
    assert FIXTURES["relevant_test_unresolved_facts"]["limitation_required"] is True


def test_unrelated_and_ambiguous_evidence_both_expect_abstention() -> None:
    assert FIXTURES["same_precedent_unrelated_proposition"]["expected_action"] == "abstain"
    assert FIXTURES["ambiguous_conflicting_evidence"]["expected_action"] == "abstain"


@pytest.mark.parametrize("fixture_id", ["relevant_test_unresolved_facts", "ahmed_salim_regression"])
def test_explanation_can_disclose_unresolved_application_without_abstaining(
    fixture_id: str,
) -> None:
    answer = GroundedAnswer.model_validate(FIXTURES[fixture_id]["answer"])

    assert answer.status is AnswerStatus.ANSWERED
    assert "does not establish whether" in answer.explanation
    assert answer.claims


def test_correct_answer_does_not_require_an_unsupported_factual_conclusion() -> None:
    answer = GroundedAnswer.model_validate(FIXTURES["relevant_test_unresolved_facts"]["answer"])
    rendered_claims = " ".join(claim.statement.casefold() for claim in answer.claims)

    assert "client satisfies" not in rendered_claims
    assert "defence is established" not in rendered_claims
    assert (
        evaluate_record(record_for(FIXTURES["relevant_test_unresolved_facts"])).validation.issues
        == ()
    )


def test_ahmed_salim_regression_answers_with_authority_and_limitation() -> None:
    fixture = FIXTURES["ahmed_salim_regression"]
    answer = GroundedAnswer.model_validate(fixture["answer"])
    outcome = evaluate_record(record_for(fixture))

    assert answer.recommended_case_id == "case:941"
    assert "three cumulative diminished-responsibility requirements" in answer.explanation
    assert "does not establish whether the present client satisfies" in answer.explanation
    assert outcome.primary_failure_layer == "1_target_present_generation_correct"


def test_prompt_text_changes_run_signature_without_changing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = package_for(FIXTURES["ahmed_salim_regression"])
    config = RAGConfig(
        modes=("facts_only", "facts_principle"),
        top_ks=(1, 3, 5),
        queries_per_stratum=16,
        seed=20260826,
        settings=settings(),
        expected_output_tokens=180,
        automatic_retries=0,
        pricing_snapshot_date="2026-08-26",
        manual_review_records=36,
    )
    evidence_before = evidence_freeze((package,))
    signature_before = benchmark_module._signature(
        config,
        (package,),
        pilot_ground_truth_digest="ground-truth-digest",
    )

    monkeypatch.setattr(benchmark_module, "SYSTEM_INSTRUCTIONS", SYSTEM_INSTRUCTIONS + " changed")

    assert (
        benchmark_module._signature(
            config,
            (package,),
            pilot_ground_truth_digest="ground-truth-digest",
        )
        != signature_before
    )
    assert evidence_freeze((package,)) == evidence_before


def test_manual_review_separates_relevance_limitation_and_factual_overreach() -> None:
    template = manual_review_template(
        [record_for(FIXTURES["ahmed_salim_regression"])], count=1, seed=20260826
    )
    review = template["records"][0]["review"]

    assert template["schema_version"] == 2
    assert review["precedent_relevance_supported"] is None
    assert review["factual_application_limit_appropriate"] is None
    assert review["unsupported_factual_conclusion_present"] is None
