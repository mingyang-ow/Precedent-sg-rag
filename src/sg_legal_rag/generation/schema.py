from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GroundedClaim(BaseModel):
    """One atomic answer claim and its verbatim support."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statement: str = Field(min_length=1, max_length=600)
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    supporting_quote: str = Field(min_length=1, max_length=700)


class GroundedAnswer(BaseModel):
    """Strict output contract shared by the provider and deterministic evaluator."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: AnswerStatus
    recommended_case_id: str | None = Field(default=None, pattern=r"^case:[0-9]+$")
    explanation: str = Field(min_length=1, max_length=1200)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_status_contract(self) -> GroundedAnswer:
        if self.status is AnswerStatus.ANSWERED:
            if self.recommended_case_id is None:
                raise ValueError("answered output requires recommended_case_id")
            if not self.claims:
                raise ValueError("answered output requires at least one cited claim")
        else:
            if self.recommended_case_id is not None:
                raise ValueError("abstention cannot recommend a case")
            if self.claims:
                raise ValueError("abstention cannot make cited claims")
        return self
