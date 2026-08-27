from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .behaviour_pilot import canonical_digest
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
from .provider import cache_path, load_record
from .schema import AnswerStatus
from .semantic_judge import (
    JUDGE_PACKAGE_VERSION,
    JUDGE_PROMPT_VERSION,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SCHEMA_VERSION,
    JUDGE_SYSTEM_INSTRUCTIONS,
    GoogleGeminiSemanticJudge,
    JudgeCallStatus,
    JudgeProviderResult,
    JudgeVerdict,
    SemanticJudgeDecision,
    SemanticJudgePackage,
    SemanticJudgeProvider,
    build_semantic_package,
    render_judge_input,
)

DEFAULT_JUDGE_CONFIG = PROJECT_ROOT / "configs" / "semantic_judge.toml"
DEFAULT_JUDGE_PACKAGES = PROJECT_ROOT / "experiments" / "samples" / "semantic_judge_pilot.json"
DEFAULT_JUDGE_REFERENCE = PROJECT_ROOT / "experiments" / "samples" / "semantic_judge_reference.json"
DEFAULT_JUDGE_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "semantic_judge_pilot.json"
DEFAULT_JUDGE_CACHE = PROJECT_ROOT / "data" / "processed" / "semantic_judge"


class JudgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["google_gemini_interactions"]
    model: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    thinking_level: Literal["low", "medium", "high"]
    max_output_tokens: int = Field(gt=0)
    expected_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, le=120)
    automatic_retries: Literal[0]
    pricing_snapshot_date: str
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_output_budget(self) -> JudgeSettings:
        if self.expected_output_tokens > self.max_output_tokens:
            raise ValueError("expected judge output cannot exceed maximum output")
        return self


class FrozenSemanticJudgePilot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    package_version: Literal["semantic-judge-package-v1"]
    source_run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")
    source_package_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_package_ids: tuple[str, ...] = Field(min_length=1)
    packages: tuple[SemanticJudgePackage, ...] = Field(min_length=1)
    package_payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_prompt_version: Literal["semantic-judge-prompt-v1"]
    judge_prompt_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_schema_version: Literal["semantic-judge-schema-v1"]
    judge_schema_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_rubric_version: Literal["semantic-grounding-rubric-v1"]
    judge_rubric_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    settings: JudgeSettings
    expected_calls: int = Field(gt=0)
    token_cost_estimate: dict[str, Any]
    run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")

    @model_validator(mode="after")
    def validate_freeze(self) -> FrozenSemanticJudgePilot:
        ids = tuple(package.source_package_id for package in self.packages)
        if ids != self.selected_package_ids or len(set(ids)) != len(ids):
            raise ValueError("semantic judge package IDs or order changed")
        if self.expected_calls != len(self.packages):
            raise ValueError("expected judge call count changed")
        payload = [package.model_dump(mode="json") for package in self.packages]
        if canonical_digest(payload) != self.package_payload_digest:
            raise ValueError("semantic judge package payload digest mismatch")
        expected_contract = judge_contract(self.settings)
        if (
            self.judge_prompt_signature != expected_contract["prompt_signature"]
            or self.judge_schema_signature != expected_contract["schema_signature"]
            or self.judge_rubric_signature != expected_contract["rubric_signature"]
        ):
            raise ValueError("semantic judge prompt, schema, or rubric changed")
        if semantic_run_signature(self) != self.run_signature:
            raise ValueError("semantic judge run signature mismatch")
        return self


class ReferenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_index: int = Field(ge=0, le=3)
    verdict: JudgeVerdict
    rationale: str = Field(min_length=1, max_length=500)


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    verdict: JudgeVerdict
    claims: tuple[ReferenceClaim, ...] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=800)


class SemanticJudgeReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    reference_version: Literal["semantic-judge-reference-v1"]
    source_run_signature: str = Field(pattern=r"^[0-9a-f]{24}$")
    source_pilot_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: tuple[str, ...] = Field(min_length=1)
    reviewer_description: str = Field(min_length=1)
    records: tuple[ReferenceRecord, ...] = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reference(self) -> SemanticJudgeReference:
        ids = [record.package_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("semantic judge reference IDs must be unique")
        if reference_digest(self) != self.digest:
            raise ValueError("semantic judge reference digest mismatch")
        return self


class JudgeExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_signature: str
    package_id: str
    package_digest: str
    result: JudgeProviderResult


def load_judge_settings(path: Path) -> JudgeSettings:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    pricing = raw["pricing"]
    return JudgeSettings.model_validate(
        raw["judge"]
        | {
            "pricing_snapshot_date": pricing["snapshot_date"],
            "input_usd_per_million": pricing["input_usd_per_million"],
            "output_usd_per_million": pricing["output_usd_per_million"],
        }
    )


def judge_contract(settings: JudgeSettings) -> dict[str, str]:
    from .semantic_judge import SEMANTIC_RUBRIC

    return {
        "prompt_signature": canonical_digest(JUDGE_SYSTEM_INSTRUCTIONS),
        "schema_signature": canonical_digest(SemanticJudgeDecision.model_json_schema()),
        "rubric_signature": canonical_digest(SEMANTIC_RUBRIC),
        "settings_signature": canonical_digest(settings.model_dump(mode="json")),
    }


def semantic_run_signature(pilot: FrozenSemanticJudgePilot) -> str:
    contract = judge_contract(pilot.settings)
    return canonical_digest(
        {
            "schema": 1,
            "source_run_signature": pilot.source_run_signature,
            "source_package_artifact_digest": pilot.source_package_artifact_digest,
            "selected_package_ids": list(pilot.selected_package_ids),
            "package_payload_digest": pilot.package_payload_digest,
            "contract": contract,
            "expected_calls": pilot.expected_calls,
        }
    )[:24]


def reference_digest(reference: SemanticJudgeReference) -> str:
    return canonical_digest(reference.model_dump(mode="json", exclude={"digest"}))


def load_frozen_judge_pilot(path: Path) -> FrozenSemanticJudgePilot:
    return FrozenSemanticJudgePilot.model_validate_json(path.read_text(encoding="utf-8"))


def load_judge_reference(path: Path) -> SemanticJudgeReference:
    return SemanticJudgeReference.model_validate_json(path.read_text(encoding="utf-8"))


def estimate_tokens_and_cost(
    packages: tuple[SemanticJudgePackage, ...], settings: JudgeSettings
) -> dict[str, Any]:
    schema = json.dumps(SemanticJudgeDecision.model_json_schema(), sort_keys=True)
    per_request = [
        math.ceil(
            len((JUDGE_SYSTEM_INSTRUCTIONS + schema + render_judge_input(package)).encode("utf-8"))
            / 4
        )
        for package in packages
    ]
    total_input = sum(per_request)
    expected_output = len(packages) * settings.expected_output_tokens
    maximum_output = len(packages) * settings.max_output_tokens
    return {
        "method": "conservative local UTF-8 bytes/4 estimate; provider accounting may differ",
        "requests": len(packages),
        "input_tokens": total_input,
        "input_tokens_per_request": {
            "min": min(per_request),
            "mean": sum(per_request) / len(per_request),
            "max": max(per_request),
        },
        "expected_output_tokens": expected_output,
        "maximum_output_tokens": maximum_output,
        "expected_cost_usd": (
            total_input * settings.input_usd_per_million
            + expected_output * settings.output_usd_per_million
        )
        / 1_000_000,
        "ceiling_cost_usd": (
            total_input * settings.input_usd_per_million
            + maximum_output * settings.output_usd_per_million
        )
        / 1_000_000,
        "pricing_snapshot_date": settings.pricing_snapshot_date,
        "pricing_usd_per_million": {
            "input": settings.input_usd_per_million,
            "output": settings.output_usd_per_million,
        },
    }


def prepare_frozen_pilot(
    *,
    judge_config_path: Path = DEFAULT_JUDGE_CONFIG,
    rag_config_path: Path = DEFAULT_CONFIG,
    global_manifest_path: Path = DEFAULT_MANIFEST,
    behaviour_manifest_path: Path = DEFAULT_BEHAVIOUR_PILOT,
    behaviour_packages_path: Path = DEFAULT_BEHAVIOUR_PACKAGES,
    behaviour_adjudication_path: Path = DEFAULT_BEHAVIOUR_ADJUDICATION,
    answer_adjudication_path: Path = DEFAULT_PILOT_ADJUDICATION,
    generation_cache_dir: Path = DEFAULT_CACHE,
) -> FrozenSemanticJudgePilot:
    settings = load_judge_settings(judge_config_path)
    rag_config, behaviour_pilot, packages, _ = preflight_frozen_behaviour_execution(
        rag_config_path=rag_config_path,
        global_manifest_path=global_manifest_path,
        behaviour_manifest_path=behaviour_manifest_path,
        behaviour_packages_path=behaviour_packages_path,
        behaviour_adjudication_path=behaviour_adjudication_path,
        answer_adjudication_path=answer_adjudication_path,
    )
    judge_packages: list[SemanticJudgePackage] = []
    for package in packages:
        record = load_record(
            cache_path(generation_cache_dir, behaviour_pilot.run_signature, package.package_id),
            run_signature=behaviour_pilot.run_signature,
            package_id=package.package_id,
        )
        if record is None:
            raise ValueError(f"historical generation cache is missing: {package.package_id}")
        assert_cached_record_matches_execution(
            record,
            package=package,
            settings=rag_config.settings,
            run_signature=behaviour_pilot.run_signature,
        )
        answer = record.result.answer
        if answer is not None and answer.status is AnswerStatus.ANSWERED:
            judge_packages.append(build_semantic_package(record))
    frozen_packages = tuple(judge_packages)
    payload_digest = canonical_digest(
        [package.model_dump(mode="json") for package in frozen_packages]
    )
    source_package_digest = canonical_digest(
        json.loads(behaviour_packages_path.read_text(encoding="utf-8"))
    )
    contract = judge_contract(settings)
    base = {
        "schema_version": 1,
        "package_version": JUDGE_PACKAGE_VERSION,
        "source_run_signature": behaviour_pilot.run_signature,
        "source_package_artifact_digest": source_package_digest,
        "selected_package_ids": tuple(package.source_package_id for package in frozen_packages),
        "packages": frozen_packages,
        "package_payload_digest": payload_digest,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_signature": contract["prompt_signature"],
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "judge_schema_signature": contract["schema_signature"],
        "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        "judge_rubric_signature": contract["rubric_signature"],
        "settings": settings,
        "expected_calls": len(frozen_packages),
        "token_cost_estimate": estimate_tokens_and_cost(frozen_packages, settings),
    }
    provisional = FrozenSemanticJudgePilot.model_construct(**base, run_signature="0" * 24)
    return FrozenSemanticJudgePilot.model_validate(
        {**base, "run_signature": semantic_run_signature(provisional)}
    )


def validate_reference_against_pilot(
    reference: SemanticJudgeReference, pilot: FrozenSemanticJudgePilot
) -> None:
    if reference.source_run_signature != pilot.source_run_signature:
        raise ValueError("semantic judge reference source run changed")
    if reference.source_pilot_digest != canonical_digest(pilot.model_dump(mode="json")):
        raise ValueError("semantic judge reference pilot digest changed")
    if tuple(record.package_id for record in reference.records) != pilot.selected_package_ids:
        raise ValueError("semantic judge reference package IDs or order changed")
    for package, record in zip(pilot.packages, reference.records, strict=True):
        expected = tuple(range(len(package.generated_answer.claims)))
        if tuple(claim.claim_index for claim in record.claims) != expected:
            raise ValueError(f"reference claim indices changed: {record.package_id}")


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_judge_results(
    *,
    pilot: FrozenSemanticJudgePilot,
    reference: SemanticJudgeReference,
    results: list[JudgeExecutionRecord],
) -> dict[str, Any]:
    validate_reference_against_pilot(reference, pilot)
    by_id = {result.package_id: result for result in results}
    if len(by_id) != len(results) or not set(by_id).issubset(pilot.selected_package_ids):
        raise ValueError("judge result IDs are duplicated or outside the frozen pilot")
    record_pairs: list[tuple[JudgeVerdict, JudgeVerdict]] = []
    claim_pairs: list[tuple[JudgeVerdict, JudgeVerdict]] = []
    disagreements: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for reference_record in reference.records:
        execution = by_id.get(reference_record.package_id)
        if execution is None or execution.result.status is JudgeCallStatus.JUDGE_UNAVAILABLE:
            unavailable.append(reference_record.package_id)
            continue
        decision = execution.result.decision
        assert decision is not None
        record_pairs.append((reference_record.verdict, decision.verdict))
        if reference_record.verdict != decision.verdict:
            disagreements.append(
                {
                    "package_id": reference_record.package_id,
                    "level": "record",
                    "claim_index": None,
                    "reference_verdict": reference_record.verdict.value,
                    "judge_verdict": decision.verdict.value,
                    "classification": "pending_manual_review",
                }
            )
        for reference_claim, judge_claim in zip(
            reference_record.claims, decision.claims, strict=True
        ):
            claim_pairs.append((reference_claim.verdict, judge_claim.verdict))
            if reference_claim.verdict != judge_claim.verdict:
                disagreements.append(
                    {
                        "package_id": reference_record.package_id,
                        "level": "claim",
                        "claim_index": reference_claim.claim_index,
                        "reference_verdict": reference_claim.verdict.value,
                        "judge_verdict": judge_claim.verdict.value,
                        "classification": "pending_manual_review",
                    }
                )

    def metrics(pairs: list[tuple[JudgeVerdict, JudgeVerdict]]) -> dict[str, Any]:
        agreement = sum(reference_value is judge_value for reference_value, judge_value in pairs)
        binary = [pair for pair in pairs if pair[0] is not JudgeVerdict.UNCERTAIN]
        tp = sum(a is JudgeVerdict.SUPPORTED and b is JudgeVerdict.SUPPORTED for a, b in binary)
        fp = sum(a is JudgeVerdict.UNSUPPORTED and b is JudgeVerdict.SUPPORTED for a, b in binary)
        fn = sum(a is JudgeVerdict.SUPPORTED and b is not JudgeVerdict.SUPPORTED for a, b in binary)
        unsupported_total = sum(a is JudgeVerdict.UNSUPPORTED for a, _ in binary)
        unsupported_detected = sum(
            a is JudgeVerdict.UNSUPPORTED and b is JudgeVerdict.UNSUPPORTED for a, b in binary
        )
        return {
            "raw_counts": {
                "evaluated": len(pairs),
                "agreement": agreement,
                "reference_uncertain_excluded_from_binary": len(pairs) - len(binary),
                "reference": dict(sorted(Counter(a.value for a, _ in pairs).items())),
                "judge": dict(sorted(Counter(b.value for _, b in pairs).items())),
            },
            "agreement": _rate(agreement, len(pairs)),
            "supported_precision": _rate(tp, tp + fp),
            "supported_recall": _rate(tp, tp + fn),
            "unsupported_detection_rate": _rate(unsupported_detected, unsupported_total),
            "uncertain_rate": _rate(
                sum(judge is JudgeVerdict.UNCERTAIN for _, judge in pairs), len(pairs)
            ),
        }

    return {
        "record_level": metrics(record_pairs),
        "claim_level": metrics(claim_pairs),
        "judge_unavailable": unavailable,
        "disagreements": disagreements,
    }


def result_cache_path(cache_dir: Path, run_signature: str, package_id: str) -> Path:
    return cache_dir / run_signature / f"{package_id}.json"


def _validate_execution_record(
    record: JudgeExecutionRecord, pilot: FrozenSemanticJudgePilot, package: SemanticJudgePackage
) -> None:
    if (
        record.run_signature != pilot.run_signature
        or record.package_id != package.source_package_id
        or record.package_digest != package.package_digest
    ):
        raise ValueError("cached semantic judge result changed")


def execute_frozen_pilot(
    *,
    pilot: FrozenSemanticJudgePilot,
    reference: SemanticJudgeReference,
    provider: SemanticJudgeProvider,
    cache_dir: Path = DEFAULT_JUDGE_CACHE,
) -> dict[str, Any]:
    validate_reference_against_pilot(reference, pilot)
    results: list[JudgeExecutionRecord] = []
    for package in pilot.packages:
        path = result_cache_path(cache_dir, pilot.run_signature, package.source_package_id)
        if path.exists():
            record = JudgeExecutionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            _validate_execution_record(record, pilot, package)
        else:
            result = provider.judge(package, pilot.settings)
            record = JudgeExecutionRecord(
                run_signature=pilot.run_signature,
                package_id=package.source_package_id,
                package_digest=package.package_digest,
                result=result,
            )
            write_json(path, record.model_dump(mode="json"))
        results.append(record)
    successful = [
        record.result for record in results if record.result.status is JudgeCallStatus.SUCCEEDED
    ]
    usage = [result.usage for result in successful if result.usage is not None]
    return {
        "schema_version": 1,
        "run_signature": pilot.run_signature,
        "provider": pilot.settings.provider,
        "model": pilot.settings.model,
        "requests": len(results),
        "automatic_retries": pilot.settings.automatic_retries,
        "operational_summary": {
            "successes": len(successful),
            "failures": len(results) - len(successful),
            "duration_seconds": sum(record.result.latency_ms for record in results) / 1000,
            "verdicts": dict(
                sorted(
                    Counter(
                        result.decision.verdict.value
                        for result in successful
                        if result.decision is not None
                    ).items()
                )
            ),
            "input_tokens": sum(item.input_tokens for item in usage),
            "output_tokens": sum(item.output_tokens for item in usage),
            "thought_tokens": sum(item.thought_tokens for item in usage),
            "estimated_cost_usd": sum(result.estimated_cost_usd or 0 for result in successful),
        },
        "metrics": evaluate_judge_results(pilot=pilot, reference=reference, results=results),
        "results": [record.model_dump(mode="json") for record in results],
    }


def _provider_from_environment(settings: JudgeSettings) -> SemanticJudgeProvider:
    api_key = os.environ.get("JUDGE_API_KEY", "")
    if not api_key:
        raise ValueError("JUDGE_API_KEY is required for paid execution")
    for name in ("OPENAI_API_KEY", "PRECEDENT_API_KEY", "PRECEDENT_METRICS_KEY"):
        if os.environ.get(name) == api_key:
            raise ValueError(f"JUDGE_API_KEY must remain separate from {name}")
    return GoogleGeminiSemanticJudge(api_key=api_key)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or execute the semantic judge pilot")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="freeze packages; no provider")
    action.add_argument("--preflight", action="store_true", help="verify the frozen paid run")
    action.add_argument("--execute", action="store_true", help="perform paid provider calls")
    parser.add_argument("--config", type=Path, default=DEFAULT_JUDGE_CONFIG)
    parser.add_argument("--packages", type=Path, default=DEFAULT_JUDGE_PACKAGES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_JUDGE_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGE_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_JUDGE_CACHE)
    parser.add_argument("--confirm-run-signature")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.prepare:
            pilot = prepare_frozen_pilot(judge_config_path=args.config)
            write_json(args.packages, pilot.model_dump(mode="json"))
            print(
                json.dumps(
                    {
                        "prepared": str(args.packages),
                        "calls": 0,
                        "run_signature": pilot.run_signature,
                    }
                )
            )
            return 0

        # All artifact, contract, reference, and confirmation checks precede provider creation.
        pilot = load_frozen_judge_pilot(args.packages)
        reference = load_judge_reference(args.reference)
        validate_reference_against_pilot(reference, pilot)
        if args.preflight:
            print(json.dumps({"verified": True, "calls": 0, "run_signature": pilot.run_signature}))
            return 0
        if args.confirm_run_signature != pilot.run_signature:
            raise ValueError("--confirm-run-signature must match the frozen paid run")
        provider = _provider_from_environment(pilot.settings)
        result = execute_frozen_pilot(
            pilot=pilot,
            reference=reference,
            provider=provider,
            cache_dir=args.cache_dir,
        )
        write_json(args.output, result)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI reports bounded preflight/provider failures.
        print(f"semantic judge failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
