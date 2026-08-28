from __future__ import annotations

import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .evidence import EvidencePackage
from .schema import GroundedClaim


class CitationValidationMode(StrEnum):
    STRICT = "strict"
    NORMALIZED = "normalized"


class CitationMatchStage(StrEnum):
    STRICT_MATCH = "strict_match"
    NFKC_MATCH = "nfkc_match"
    WHITESPACE_MATCH = "whitespace_match"
    MOJIBAKE_NORMALIZED_MATCH = "mojibake_normalized_match"
    NO_MATCH = "no_match"


OBSERVED_MOJIBAKE_EQUIVALENTS = {
    "â\x80\x93": "–",
    "â\x80\x99": "’",
}


class CitationMatchAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    citation_target_valid: bool
    authority_supplied_in_evidence: bool
    historical_strict_match: bool
    normalized_match: bool
    earliest_match_stage: CitationMatchStage


def citation_matches(audit: CitationMatchAudit, mode: CitationValidationMode) -> bool:
    if mode is CitationValidationMode.STRICT:
        return audit.historical_strict_match
    return audit.normalized_match


def normalize_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_observed_mojibake(text: str) -> str:
    normalized = text
    for corrupted, canonical in OBSERVED_MOJIBAKE_EQUIVALENTS.items():
        normalized = normalized.replace(corrupted, canonical)
    return normalized


def historical_strict_match(quote: str, passage: str) -> bool:
    """Preserve the existing evaluator's combined NFKC/whitespace comparison."""

    return normalize_whitespace(normalize_nfkc(quote)) in normalize_whitespace(
        normalize_nfkc(passage)
    )


def citation_match_stage(quote: str, passage: str) -> CitationMatchStage:
    if quote in passage:
        return CitationMatchStage.STRICT_MATCH

    nfkc_quote = normalize_nfkc(quote)
    nfkc_passage = normalize_nfkc(passage)
    if nfkc_quote in nfkc_passage:
        return CitationMatchStage.NFKC_MATCH

    whitespace_quote = normalize_whitespace(nfkc_quote)
    whitespace_passage = normalize_whitespace(nfkc_passage)
    if whitespace_quote in whitespace_passage:
        return CitationMatchStage.WHITESPACE_MATCH

    mojibake_quote = normalize_observed_mojibake(whitespace_quote)
    mojibake_passage = normalize_observed_mojibake(whitespace_passage)
    if mojibake_quote in mojibake_passage:
        return CitationMatchStage.MOJIBAKE_NORMALIZED_MATCH
    return CitationMatchStage.NO_MATCH


def audit_claim_citation(
    package: EvidencePackage,
    claim: GroundedClaim,
    *,
    recommended_case_id: str | None,
) -> CitationMatchAudit:
    evidence_by_id = {item.evidence_id: item for item in package.evidence}
    evidence = evidence_by_id.get(claim.evidence_id)
    supplied_case_ids = {item.case_id for item in package.evidence}
    authority_supplied = (
        recommended_case_id is not None and recommended_case_id in supplied_case_ids
    )
    if evidence is None:
        return CitationMatchAudit(
            evidence_id=claim.evidence_id,
            citation_target_valid=False,
            authority_supplied_in_evidence=authority_supplied,
            historical_strict_match=False,
            normalized_match=False,
            earliest_match_stage=CitationMatchStage.NO_MATCH,
        )
    stage = citation_match_stage(claim.supporting_quote, evidence.passage)
    strict = historical_strict_match(claim.supporting_quote, evidence.passage)
    return CitationMatchAudit(
        evidence_id=claim.evidence_id,
        citation_target_valid=True,
        authority_supplied_in_evidence=authority_supplied,
        historical_strict_match=strict,
        normalized_match=(strict or stage is CitationMatchStage.MOJIBAKE_NORMALIZED_MATCH),
        earliest_match_stage=stage,
    )
