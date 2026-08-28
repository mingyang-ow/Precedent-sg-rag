from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from sg_legal_rag.generation import production_contract as contract_module
from sg_legal_rag.generation.behaviour_pilot import load_frozen_behaviour_packages
from sg_legal_rag.generation.benchmark import (
    DEFAULT_BEHAVIOUR_PACKAGES,
    package_evidence_lock,
)
from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidenceItem,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_CITATION_CONTRACT_VERSION,
    PRODUCTION_PROMPT_VERSION,
    PRODUCTION_SYSTEM_INSTRUCTIONS,
    CitationContractIssueCode,
    CitationContractViolation,
    FrozenEvidenceResolver,
    ProductionAnswer,
    ProductionClaim,
    parse_and_resolve_production_answer,
    production_prompt_signature,
    production_schema_signature,
    resolve_production_answer,
)
from sg_legal_rag.generation.provider import SYSTEM_INSTRUCTIONS
from sg_legal_rag.generation.schema import AnswerStatus, GroundedAnswer


def evidence_item(
    evidence_id: str = "E1",
    *,
    case_id: str = "case:941",
    passage: str = "The court identified three cumulative requirements.",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        case_id=case_id,
        case_name="Ahmed Salim v Public Prosecutor",
        source_judgment="[2024] SGCA 1",
        source_url="https://example.test/judgment",
        source_year=2024,
        passage=passage,
        passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        retrieval_rank=1,
        retrieval_score=12.5,
        origin=EvidenceOrigin.HISTORICAL_RETRIEVAL,
        gold_row_id=None,
        citation_relationship_verified=True,
    )


@pytest.fixture
def package() -> EvidencePackage:
    return EvidencePackage(
        package_id="production-package",
        query_id="production-query",
        query_mode="facts_only",
        query_text="What authority states the diminished-responsibility requirements?",
        stratum="production_fixture",
        condition=EvidenceCondition.RETRIEVED,
        top_k=1,
        evidence=(evidence_item(),),
        accepted_case_ids=("case:941",),
        warm_start=True,
        target_present=True,
        evidence_sufficient=True,
        expected_action=ExpectedAction.ANSWER,
        sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT,
    )


def answered(*claims: ProductionClaim, recommendation: str = "case:941") -> ProductionAnswer:
    return ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.ANSWERED,
        recommended_case_id=recommendation,
        explanation="The authority states the test; application to the present facts is unresolved.",
        claims=claims
        or (
            ProductionClaim(
                statement="The authority identifies three cumulative requirements.",
                evidence_id="E1",
                case_id="case:941",
            ),
        ),
    )


def issue_codes(error: CitationContractViolation) -> set[CitationContractIssueCode]:
    return {issue.code for issue in error.issues}


def test_valid_evidence_reference_resolves_exact_application_owned_source(
    package: EvidencePackage,
) -> None:
    source = package.evidence[0]

    resolved = resolve_production_answer(package, answered())
    citation = resolved.claims[0].citation

    assert resolved.package_id == package.package_id
    assert resolved.query_id == package.query_id
    assert citation.evidence_id == source.evidence_id
    assert citation.case_id == source.case_id
    assert citation.case_name == source.case_name
    assert citation.source_text == source.passage
    assert citation.source_judgment == source.source_judgment
    assert citation.source_year == source.source_year
    assert citation.passage_digest == source.passage_digest
    assert citation.retrieval_rank == source.retrieval_rank
    assert citation.retrieval_score == source.retrieval_score


def test_known_evidence_absent_from_model_context_is_rejected() -> None:
    first = evidence_item()
    second = evidence_item(
        "E2", case_id="case:942", passage="A second passage not supplied to the model."
    )
    resolver = FrozenEvidenceResolver(
        package_id="context",
        query_id="query",
        evidence=(first, second),
        visible_evidence_ids=frozenset({"E1"}),
    )
    answer = answered(
        ProductionClaim(
            statement="The hidden passage supplies another proposition.",
            evidence_id="E2",
            case_id="case:942",
        )
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolver.resolve(answer)

    assert issue_codes(captured.value) == {CitationContractIssueCode.EVIDENCE_NOT_SUPPLIED}


def test_hallucinated_evidence_id_is_rejected(package: EvidencePackage) -> None:
    answer = answered(
        ProductionClaim(
            statement="A hallucinated source supposedly supports this.",
            evidence_id="E99",
            case_id="case:941",
        )
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answer)

    assert issue_codes(captured.value) == {CitationContractIssueCode.UNKNOWN_EVIDENCE_ID}


def test_case_and_evidence_mismatch_is_rejected(package: EvidencePackage) -> None:
    answer = answered(
        ProductionClaim(
            statement="The cited evidence belongs to a different case.",
            evidence_id="E1",
            case_id="case:942",
        )
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answer)

    assert issue_codes(captured.value) == {CitationContractIssueCode.CASE_EVIDENCE_MISMATCH}


def test_model_cannot_supply_source_text(package: EvidencePackage) -> None:
    payload = answered().model_dump(mode="json")
    payload["claims"][0]["source_text"] = "Model-controlled replacement passage."

    with pytest.raises(CitationContractViolation) as captured:
        parse_and_resolve_production_answer(package, payload)

    assert issue_codes(captured.value) == {CitationContractIssueCode.MALFORMED_STRUCTURED_OUTPUT}


def test_duplicate_evidence_references_are_rejected(package: EvidencePackage) -> None:
    first = ProductionClaim(statement="First claim.", evidence_id="E1", case_id="case:941")
    second = ProductionClaim(statement="Second claim.", evidence_id="E1", case_id="case:941")

    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answered(first, second))

    assert CitationContractIssueCode.DUPLICATE_EVIDENCE_REFERENCE in issue_codes(captured.value)


def test_unsupplied_recommendation_is_rejected(package: EvidencePackage) -> None:
    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answered(recommendation="case:999"))

    assert CitationContractIssueCode.UNSUPPLIED_RECOMMENDATION in issue_codes(captured.value)


def test_answer_without_supporting_evidence_is_rejected(package: EvidencePackage) -> None:
    answer = ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.ANSWERED,
        recommended_case_id="case:941",
        explanation="An uncited answer.",
        claims=(),
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answer)

    assert issue_codes(captured.value) == {
        CitationContractIssueCode.ANSWER_WITHOUT_SUPPORTING_EVIDENCE
    }


def test_abstention_resolves_without_citations(package: EvidencePackage) -> None:
    answer = ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        recommended_case_id=None,
        explanation="The supplied evidence does not support a precedent recommendation.",
        claims=(),
    )

    resolved = resolve_production_answer(package, answer)

    assert resolved.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert resolved.recommended_case_id is None
    assert resolved.claims == ()


def test_abstention_with_claim_citations_is_rejected(package: EvidencePackage) -> None:
    answer = ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        recommended_case_id=None,
        explanation="Invalid abstention with a claim.",
        claims=(ProductionClaim(statement="A cited claim.", evidence_id="E1", case_id="case:941"),),
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolve_production_answer(package, answer)

    assert CitationContractIssueCode.ABSTENTION_WITH_CITATIONS in issue_codes(captured.value)


def test_malformed_structured_output_is_rejected(package: EvidencePackage) -> None:
    with pytest.raises(CitationContractViolation) as captured:
        parse_and_resolve_production_answer(package, "{not valid JSON")

    assert issue_codes(captured.value) == {CitationContractIssueCode.MALFORMED_STRUCTURED_OUTPUT}


def test_changed_stored_passage_digest_fails_integrity_check(package: EvidencePackage) -> None:
    corrupt = package.evidence[0].model_copy(update={"passage_digest": "0" * 64})
    resolver = FrozenEvidenceResolver(
        package_id=package.package_id,
        query_id=package.query_id,
        evidence=(corrupt,),
        visible_evidence_ids=frozenset({"E1"}),
    )

    with pytest.raises(CitationContractViolation) as captured:
        resolver.resolve(answered())

    assert issue_codes(captured.value) == {CitationContractIssueCode.EVIDENCE_DIGEST_MISMATCH}


def test_resolution_leaves_frozen_historical_package_and_signature_unchanged() -> None:
    package = load_frozen_behaviour_packages(DEFAULT_BEHAVIOUR_PACKAGES).packages[0]
    evidence = package.evidence[0]
    package_before = package.model_dump_json()
    signature_before = package_evidence_lock(package)
    answer = ProductionAnswer(
        contract_version=PRODUCTION_CITATION_CONTRACT_VERSION,
        status=AnswerStatus.ANSWERED,
        recommended_case_id=evidence.case_id,
        explanation="The stored passage supplies a bounded proposition.",
        claims=(
            ProductionClaim(
                statement="The claim is traceable to frozen evidence.",
                evidence_id=evidence.evidence_id,
                case_id=evidence.case_id,
            ),
        ),
    )

    resolved = resolve_production_answer(package, answer)

    assert resolved.claims[0].citation.source_text == evidence.passage
    assert package.model_dump_json() == package_before
    assert package_evidence_lock(package) == signature_before


def test_historical_schema_remains_loadable_and_signature_is_unchanged() -> None:
    historical_payload = {
        "status": "answered",
        "recommended_case_id": "case:941",
        "explanation": "Historical output with a generated quotation.",
        "claims": [
            {
                "statement": "The old contract required a quote.",
                "evidence_id": "E1",
                "supporting_quote": "an exact historical quote",
            }
        ],
    }

    historical = GroundedAnswer.model_validate(historical_payload)
    signature_payload = json.dumps(
        GroundedAnswer.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert historical.claims[0].supporting_quote == "an exact historical quote"
    assert hashlib.sha256(signature_payload.encode("utf-8")).hexdigest() == (
        "61e54fb6213abad2a8975479641a6e5f9b19e44361c826061cbfbb856bb87eeb"
    )
    with pytest.raises(ValidationError):
        ProductionAnswer.model_validate(historical_payload)


def test_production_schema_version_and_signature_are_deterministic() -> None:
    assert PRODUCTION_CITATION_CONTRACT_VERSION == "production-citation-v1"
    assert production_schema_signature() == (
        "4ca7a25a7860f782ce176890498593e50144ee26eb3eb11b16fd3b82c091be65"
    )


def test_production_prompt_preserves_answerability_and_removes_quote_generation() -> None:
    normalized = " ".join(PRODUCTION_SYSTEM_INSTRUCTIONS.split())

    assert PRODUCTION_PROMPT_VERSION == "rag-production-v1"
    assert "relevant authority for the legal principle, rule, or test" in normalized
    assert "even if factual application remains unresolved" in normalized
    assert "Abstain only when" in normalized
    assert "Do not generate, copy, or paraphrase source_text" in normalized
    assert production_prompt_signature() == (
        "91557588b2157efa9aca3d6c9a5d75a1fb41ed1f0b4b3005a7a37c9200f7328a"
    )
    assert hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest() == (
        "29fa06887d945fd91959c89b6d9637d0cb732beb21ae4f5d2bd001aa9e3446be"
    )


def test_production_contract_does_not_construct_a_provider() -> None:
    assert "OpenAIResponsesGenerator" not in contract_module.__dict__
