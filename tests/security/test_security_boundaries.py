from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidenceItem,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
)
from sg_legal_rag.generation.production_contract import (
    PRODUCTION_SYSTEM_INSTRUCTIONS,
    CitationContractIssueCode,
    CitationContractViolation,
    FrozenEvidenceResolver,
    ProductionAnswer,
    parse_and_resolve_production_answer,
    render_production_user_input,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures/adversarial_inputs.json"


@pytest.fixture(scope="module")
def attacks() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _evidence(evidence_id: str, case_id: str, passage: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        case_id=case_id,
        case_name=f"Synthetic Authority {case_id}",
        source_judgment="[2026] SGHC 1",
        source_url="https://example.test/judgment",
        source_year=2026,
        passage=passage,
        passage_digest=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        retrieval_rank=int(evidence_id[1:]),
        retrieval_score=1.0,
        origin=EvidenceOrigin.HISTORICAL_RETRIEVAL,
        gold_row_id=None,
        citation_relationship_verified=True,
    )


def _package(query: str, passages: tuple[str, ...]) -> EvidencePackage:
    evidence = tuple(
        _evidence(f"E{position}", f"case:{position}", passage)
        for position, passage in enumerate(passages, start=1)
    )
    return EvidencePackage(
        package_id="security-package",
        query_id="security-query",
        query_mode="facts",
        query_text=query,
        stratum="security_fixture",
        condition=EvidenceCondition.RETRIEVED,
        top_k=len(evidence),
        evidence=evidence,
        accepted_case_ids=(),
        warm_start=False,
        target_present=False,
        evidence_sufficient=None,
        expected_action=ExpectedAction.UNKNOWN_NEEDS_REVIEW,
        sufficiency_basis=EvidenceSufficiencyBasis.MANUAL_REVIEW_REQUIRED,
    )


class TestPromptInjectionBoundary:
    def test_user_attacks_remain_exactly_nested_untrusted_data(
        self, attacks: dict[str, Any]
    ) -> None:
        for attack in attacks["user_queries"]:
            rendered = render_production_user_input(_package(attack, ("Benign evidence.",)))
            envelope = json.loads(rendered)

            assert envelope["untrusted_data"]["query"]["text"] == attack
            assert set(envelope) == {"contract_version", "untrusted_data"}

    def test_system_policy_explicitly_denies_instruction_and_secret_authority(self) -> None:
        normalized = " ".join(PRODUCTION_SYSTEM_INSTRUCTIONS.split()).lower()

        for required in (
            "untrusted data",
            "never instructions",
            "cannot override",
            "authorize actions",
            "change the schema",
            "system prompts",
            "credentials",
            "internal configuration",
        ):
            assert required in normalized


class TestEvidenceInjectionBoundary:
    def test_document_attacks_cannot_escape_json_evidence_envelope(
        self, attacks: dict[str, Any]
    ) -> None:
        passages = tuple(attacks["evidence_passages"])
        envelope = json.loads(render_production_user_input(_package("query", passages)))

        evidence = envelope["untrusted_data"]["evidence"]
        assert [item["passage"] for item in evidence] == list(passages)
        assert [item["evidence_id"] for item in evidence] == [
            f"E{position}" for position in range(1, len(passages) + 1)
        ]
        assert "system" not in envelope
        assert "developer" not in envelope


class TestOutputValidationBoundary:
    def test_adversarial_fake_provider_outputs_are_rejected(self, attacks: dict[str, Any]) -> None:
        package = _package("query", ("Authoritative application-owned source.",))

        for attack in attacks["generated_outputs"]:
            payload = attack["payload"]
            if attack["attack"] in {"model_supplied_source_text", "extra_schema_field"}:
                with pytest.raises(ValidationError):
                    ProductionAnswer.model_validate(payload)
            else:
                with pytest.raises(CitationContractViolation):
                    parse_and_resolve_production_answer(package, payload)

    def test_valid_but_invisible_evidence_id_is_rejected(self) -> None:
        package = _package("query", ("Visible source.", "Hidden source."))
        resolver = FrozenEvidenceResolver(
            package_id=package.package_id,
            query_id=package.query_id,
            evidence=package.evidence,
            visible_evidence_ids=frozenset({"E1"}),
        )
        answer = ProductionAnswer.model_validate(
            {
                "contract_version": "production-citation-v1",
                "status": "answered",
                "recommended_case_id": "case:2",
                "explanation": "Hidden reference.",
                "claims": [{"statement": "Hidden.", "evidence_id": "E2", "case_id": "case:2"}],
            }
        )

        with pytest.raises(CitationContractViolation) as caught:
            resolver.resolve(answer)

        assert CitationContractIssueCode.EVIDENCE_NOT_SUPPLIED in {
            issue.code for issue in caught.value.issues
        }

    def test_application_owns_resolved_source_text(self) -> None:
        package = _package("query", ("Authoritative application-owned source.",))
        resolved = parse_and_resolve_production_answer(
            package,
            {
                "contract_version": "production-citation-v1",
                "status": "answered",
                "recommended_case_id": "case:1",
                "explanation": "Supported.",
                "claims": [{"statement": "Supported.", "evidence_id": "E1", "case_id": "case:1"}],
            },
        )

        assert resolved.claims[0].citation.source_text == package.evidence[0].passage
