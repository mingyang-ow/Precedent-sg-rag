from __future__ import annotations

import json
from pathlib import Path

from sg_legal_rag.generation import citation_audit as audit_module
from sg_legal_rag.generation.behaviour_pilot import load_frozen_behaviour_packages
from sg_legal_rag.generation.benchmark import (
    DEFAULT_BEHAVIOUR_PACKAGES,
    package_evidence_lock,
)
from sg_legal_rag.generation.citation_audit import (
    DEFAULT_MANUAL_AUDIT,
    load_manual_audit,
    manual_audit_digest,
)
from sg_legal_rag.generation.citation_validation import (
    OBSERVED_MOJIBAKE_EQUIVALENTS,
    CitationMatchStage,
    CitationValidationMode,
    audit_claim_citation,
    citation_match_stage,
    citation_matches,
    historical_strict_match,
    normalize_observed_mojibake,
)
from sg_legal_rag.generation.evaluation import normalize_space
from sg_legal_rag.generation.schema import GroundedClaim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITATION_RESULT = PROJECT_ROOT / "experiments/results/rag_baseline_behaviour_citation_audit.json"
CLEANROOM_RESULT = PROJECT_ROOT / "experiments/results/rag_baseline_behaviour_cleanroom.json"


def test_historical_strict_mode_remains_the_existing_evaluator_comparison() -> None:
    examples = (
        ("same text", "same text"),
        ("fullwidth A", "fullwidth Ａ"),
        ("space separated", "space\n separated"),
        ("not present", "different"),
    )

    for quote, passage in examples:
        assert historical_strict_match(quote, passage) == (
            normalize_space(quote) in normalize_space(passage)
        )


def test_nfkc_stage_is_reported_independently() -> None:
    assert citation_match_stage("fullwidth A", "fullwidth Ａ") is CitationMatchStage.NFKC_MATCH


def test_whitespace_stage_is_reported_independently() -> None:
    assert (
        citation_match_stage("space separated", "space\n\t separated")
        is CitationMatchStage.WHITESPACE_MATCH
    )


def test_observed_en_dash_mojibake_mapping() -> None:
    assert OBSERVED_MOJIBAKE_EQUIVALENTS["â\x80\x93"] == "–"
    assert normalize_observed_mojibake("alpha â\x80\x93 beta") == "alpha – beta"
    assert (
        citation_match_stage("alpha – beta", "alpha â\x80\x93 beta")
        is CitationMatchStage.MOJIBAKE_NORMALIZED_MATCH
    )


def test_observed_right_quote_mojibake_mapping() -> None:
    assert OBSERVED_MOJIBAKE_EQUIVALENTS["â\x80\x99"] == "’"
    assert normalize_observed_mojibake("accusedâ\x80\x99s") == "accused’s"
    assert (
        citation_match_stage("accused’s", "accusedâ\x80\x99s")
        is CitationMatchStage.MOJIBAKE_NORMALIZED_MATCH
    )


def test_unobserved_mojibake_and_fuzzy_variants_remain_failures() -> None:
    assert (
        citation_match_stage("alpha — beta", "alpha â\x80\x94 beta") is CitationMatchStage.NO_MATCH
    )
    assert citation_match_stage("color", "colour") is CitationMatchStage.NO_MATCH
    assert (
        citation_match_stage("three exact words", "three ... words") is CitationMatchStage.NO_MATCH
    )


def test_normalized_matching_does_not_mutate_output_evidence_or_signature() -> None:
    package = load_frozen_behaviour_packages(DEFAULT_BEHAVIOUR_PACKAGES).packages[0]
    claim = GroundedClaim(
        statement="Synthetic audit claim.",
        evidence_id=package.evidence[0].evidence_id,
        supporting_quote=package.evidence[0].passage[:40],
    )
    package_before = package.model_dump_json()
    claim_before = claim.model_dump_json()
    signature_before = package_evidence_lock(package)

    audit = audit_claim_citation(
        package,
        claim,
        recommended_case_id=package.evidence[0].case_id,
    )

    assert audit.historical_strict_match
    assert citation_matches(audit, CitationValidationMode.STRICT)
    assert citation_matches(audit, CitationValidationMode.NORMALIZED)
    assert package.model_dump_json() == package_before
    assert claim.model_dump_json() == claim_before
    assert package_evidence_lock(package) == signature_before


def test_strict_and_normalized_metrics_coexist_without_overwrite() -> None:
    audit = json.loads(CITATION_RESULT.read_text(encoding="utf-8"))
    cleanroom = json.loads(CLEANROOM_RESULT.read_text(encoding="utf-8"))

    assert audit["strict_mode_preserved"] is True
    assert audit["model_visible_inputs_changed"] is False
    assert audit["historical_run_signature"] == "3664b44b7d4dbe620225d598"
    assert audit["historical_evidence_digest"] == (
        "9faf464cb462aa3a4b87a13942f7bea4f7c81cba6db99a97ea8a165aca5cebb5"
    )
    assert audit["historical_generation_contract"]["model"] == "gpt-5.6-luna"
    assert audit["historical_generation_contract"]["prompt_signature"] == (
        "29fa06887d945fd91959c89b6d9637d0cb732beb21ae4f5d2bd001aa9e3446be"
    )
    assert audit["historical_generation_contract"]["output_schema_signature"] == (
        "61e54fb6213abad2a8975479641a6e5f9b19e44361c826061cbfbb856bb87eeb"
    )
    assert audit["metrics"]["strict"] == {
        "answered_records": 8,
        "citation_completeness": 1.0,
        "citation_correctness": 0.4375,
        "citation_validity": 0.375,
        "fully_valid_answered_records": 3,
        "unsupported_claim_rate": 0.5625,
    }
    assert audit["metrics"]["normalized"]["citation_validity"] == 0.625
    assert cleanroom["output_quality_all_12"]["citation_validity"] == 0.375


def test_manual_audit_digest_and_coverage_are_frozen() -> None:
    audit = load_manual_audit(DEFAULT_MANUAL_AUDIT)

    assert manual_audit_digest(audit) == audit.digest
    assert audit.digest == "7ae9744500afb1d1da4d76b3e322bfe207bdd0146ac5dd526da9f89b9e0cbb9a"
    assert len(audit.records) == 7


def test_citation_audit_has_no_provider_constructor() -> None:
    assert "OpenAIResponsesGenerator" not in audit_module.__dict__
