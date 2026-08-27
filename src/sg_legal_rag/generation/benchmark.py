from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sg_legal_rag.retrieval.benchmark import DEFAULT_DATA_DIR, DEFAULT_SPLITS
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import load_corpus_repair_dataset
from sg_legal_rag.retrieval.corpus_repair_benchmark import (
    DEFAULT_CONFIG as DEFAULT_CORPUS_CONFIG,
)
from sg_legal_rag.retrieval.corpus_repair_benchmark import (
    load_config as load_corpus_config,
)

from .adjudication import (
    PilotAdjudication,
    adjudication_digest,
    apply_adjudication,
    load_pilot_adjudication,
)
from .behaviour_pilot import (
    BehaviourAdjudication,
    BehaviourPilotManifest,
    FrozenBehaviourPackages,
    apply_behaviour_labels,
    behaviour_adjudication_digest,
    behaviour_run_signature,
    canonical_digest,
    load_behaviour_adjudication,
    load_behaviour_pilot,
    load_frozen_behaviour_packages,
    validate_behaviour_pilot,
)
from .evaluation import evaluate_record, grouped_summaries
from .evidence import (
    EvidenceCondition,
    EvidenceOrigin,
    EvidencePackage,
    ExpectedAction,
    prompt_evidence,
)
from .provider import (
    SYSTEM_INSTRUCTIONS,
    GenerationRecord,
    GenerationSettings,
    OpenAIResponsesGenerator,
    ProviderCallStatus,
    cache_path,
    generate_record,
    load_record,
    render_user_input,
    save_record,
)
from .sampling import COLD, WARM_FAILURE, WARM_SUCCESS, build_packages, select_queries
from .schema import GroundedAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rag_baseline.toml"
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "samples" / "rag_baseline.json"
DEFAULT_PILOT_ADJUDICATION = (
    PROJECT_ROOT / "experiments" / "samples" / "rag_pilot_adjudication.json"
)
DEFAULT_BEHAVIOUR_ADJUDICATION = (
    PROJECT_ROOT / "experiments" / "samples" / "rag_behaviour_adjudication.json"
)
DEFAULT_BEHAVIOUR_PILOT = PROJECT_ROOT / "experiments" / "samples" / "rag_behaviour_pilot.json"
DEFAULT_BEHAVIOUR_PACKAGES = (
    PROJECT_ROOT / "experiments" / "samples" / "rag_behaviour_packages.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "rag_baseline.json"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "processed" / "generation"
DEFAULT_MANUAL_REVIEW = PROJECT_ROOT / "data" / "processed" / "rag_manual_review.json"
DEFAULT_SUFFICIENCY_REVIEW = PROJECT_ROOT / "data" / "processed" / "rag_sufficiency_review.json"
TOKEN_OVERHEAD_PER_REQUEST = 40


@dataclass(frozen=True)
class RAGConfig:
    modes: tuple[str, ...]
    top_ks: tuple[int, ...]
    queries_per_stratum: int
    seed: int
    settings: GenerationSettings
    expected_output_tokens: int
    automatic_retries: int
    pricing_snapshot_date: str
    manual_review_records: int


def load_config(path: Path) -> RAGConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    generation = raw["generation"]
    pricing = raw["pricing"]
    if generation["provider"] != "openai_responses":
        raise ValueError("the Phase 3 baseline supports only openai_responses")
    settings = GenerationSettings(
        model=str(generation["model"]),
        reasoning_effort=str(generation["reasoning_effort"]),
        verbosity=str(generation["verbosity"]),
        max_output_tokens=int(generation["max_output_tokens"]),
        prompt_version=str(generation["prompt_version"]),
        temperature=None,
        seed=None,
        input_usd_per_million=float(pricing["input_usd_per_million"]),
        cached_input_usd_per_million=float(pricing["cached_input_usd_per_million"]),
        output_usd_per_million=float(pricing["output_usd_per_million"]),
    )
    config = RAGConfig(
        modes=tuple(str(value) for value in raw["sampling"]["modes"]),
        top_ks=tuple(int(value) for value in raw["sampling"]["top_ks"]),
        queries_per_stratum=int(raw["sampling"]["queries_per_stratum"]),
        seed=int(raw["sampling"]["seed"]),
        settings=settings,
        expected_output_tokens=int(generation["expected_output_tokens"]),
        automatic_retries=int(generation["automatic_retries"]),
        pricing_snapshot_date=str(pricing["snapshot_date"]),
        manual_review_records=int(raw["evaluation"]["manual_review_records"]),
    )
    if not config.modes or len(set(config.modes)) != len(config.modes):
        raise ValueError("sampling modes must be non-empty and unique")
    if tuple(sorted(set(config.top_ks))) != config.top_ks or any(
        value < 1 for value in config.top_ks
    ):
        raise ValueError("top_ks must be positive, unique, and increasing")
    if (
        min(
            config.queries_per_stratum,
            config.expected_output_tokens,
            config.manual_review_records,
        )
        < 1
    ):
        raise ValueError("sampling and evaluation counts must be positive")
    if config.expected_output_tokens > settings.max_output_tokens:
        raise ValueError("expected output tokens cannot exceed max_output_tokens")
    if config.automatic_retries != 0:
        raise ValueError("automatic retries must remain disabled for bounded cost")
    return config


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_evidence_lock(package: EvidencePackage) -> dict[str, Any]:
    visible_evidence = prompt_evidence(package)
    return {
        "package_id": package.package_id,
        "query_id": package.query_id,
        "condition": package.condition.value,
        "top_k": package.top_k,
        "target_present": package.target_present,
        "evidence_sufficient": package.evidence_sufficient,
        "expected_action": package.expected_action.value,
        "sufficiency_basis": package.sufficiency_basis.value,
        "evidence_digests": [
            {
                "evidence_id": item.evidence_id,
                "case_id": item.case_id,
                "passage_digest": item.passage_digest,
                "origin": item.origin.value,
                "gold_row_id": item.gold_row_id,
                "citation_relationship_verified": item.citation_relationship_verified,
            }
            for item in package.evidence
        ],
        "evidence_signature": _canonical_digest(visible_evidence),
        "input_signature": hashlib.sha256(render_user_input(package).encode("utf-8")).hexdigest(),
    }


def evidence_freeze(packages: tuple[EvidencePackage, ...]) -> dict[str, Any]:
    locks = [package_evidence_lock(package) for package in packages]
    return {
        "algorithm": "sha256-canonical-json-v2",
        "signature": _canonical_digest(locks),
        "packages": locks,
    }


def _config_signature_payload(config: RAGConfig) -> dict[str, Any]:
    return {
        "modes": list(config.modes),
        "top_ks": list(config.top_ks),
        "queries_per_stratum": config.queries_per_stratum,
        "seed": config.seed,
        "settings": config.settings.model_dump(mode="json"),
        "expected_output_tokens": config.expected_output_tokens,
        "automatic_retries": config.automatic_retries,
        "pricing_snapshot_date": config.pricing_snapshot_date,
        "manual_review_records": config.manual_review_records,
        "prompt_signature": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
    }


def _signature_from_evidence_signature(
    config: RAGConfig,
    evidence_signature: str,
    *,
    pilot_ground_truth_digest: str,
) -> str:
    payload = {
        "cache_schema": 4,
        "config": _config_signature_payload(config),
        "evidence_signature": evidence_signature,
        "pilot_ground_truth_digest": pilot_ground_truth_digest,
    }
    return _canonical_digest(payload)[:24]


def _signature(
    config: RAGConfig,
    packages: tuple[EvidencePackage, ...],
    *,
    pilot_ground_truth_digest: str,
) -> str:
    frozen_evidence = evidence_freeze(packages)
    return _signature_from_evidence_signature(
        config,
        frozen_evidence["signature"],
        pilot_ground_truth_digest=pilot_ground_truth_digest,
    )


def _tokenizer(model: str) -> tuple[Any, str]:
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model), f"tiktoken:model:{model}"
    except KeyError:
        return tiktoken.get_encoding("o200k_base"), "tiktoken:o200k_base-fallback"


def estimate_tokens_and_cost(
    packages: tuple[EvidencePackage, ...], config: RAGConfig
) -> dict[str, Any]:
    encoding, tokenizer_name = _tokenizer(config.settings.model)
    schema = json.dumps(GroundedAnswer.model_json_schema(), sort_keys=True)
    shared_tokens = len(encoding.encode(SYSTEM_INSTRUCTIONS)) + len(encoding.encode(schema))
    per_request_input = [
        shared_tokens
        + len(encoding.encode(render_user_input(package)))
        + TOKEN_OVERHEAD_PER_REQUEST
        for package in packages
    ]
    total_input = sum(per_request_input)
    expected_output = len(packages) * config.expected_output_tokens
    maximum_output = len(packages) * config.settings.max_output_tokens
    expected_cost = (
        total_input * config.settings.input_usd_per_million
        + expected_output * config.settings.output_usd_per_million
    ) / 1_000_000
    maximum_cost = (
        total_input * config.settings.input_usd_per_million
        + maximum_output * config.settings.output_usd_per_million
    ) / 1_000_000
    return {
        "method": (
            "local tokenizer estimate including instructions, JSON schema, prompt, and fixed "
            "request overhead; server accounting may differ"
        ),
        "tokenizer": tokenizer_name,
        "requests": len(packages),
        "input_tokens": total_input,
        "input_tokens_per_request": {
            "min": min(per_request_input),
            "mean": sum(per_request_input) / len(per_request_input),
            "max": max(per_request_input),
        },
        "expected_output_tokens": expected_output,
        "maximum_output_tokens": maximum_output,
        "expected_cost_usd_no_cache": expected_cost,
        "maximum_cost_usd_no_cache": maximum_cost,
        "pricing_snapshot_date": config.pricing_snapshot_date,
        "pricing_usd_per_million": {
            "input": config.settings.input_usd_per_million,
            "cached_input": config.settings.cached_input_usd_per_million,
            "output": config.settings.output_usd_per_million,
        },
    }


def select_pilot(packages: tuple[EvidencePackage, ...]) -> tuple[EvidencePackage, ...]:
    """Two-mode, 12-call pilot spanning oracle, retrieval strata, and k endpoints."""

    chosen: list[EvidencePackage] = []
    modes = sorted({package.query_mode for package in packages})
    for mode in modes:
        mode_packages = [package for package in packages if package.query_mode == mode]
        predicates = (
            lambda p: (
                p.condition is EvidenceCondition.ORACLE_GOLD
                and p.stratum == WARM_SUCCESS
                and p.expected_action is ExpectedAction.ANSWER
            ),
            lambda p: (
                p.condition is EvidenceCondition.RETRIEVED and p.target_present and p.top_k == 1
            ),
            lambda p: (
                p.condition is EvidenceCondition.RETRIEVED and p.target_present and p.top_k == 5
            ),
            lambda p: (
                p.condition is EvidenceCondition.RETRIEVED
                and not p.target_present
                and p.stratum == WARM_FAILURE
                and p.top_k == 5
            ),
            lambda p: (
                p.condition is EvidenceCondition.RETRIEVED
                and not p.target_present
                and p.stratum == COLD
                and p.top_k == 1
            ),
            lambda p: (
                p.condition is EvidenceCondition.RETRIEVED
                and not p.target_present
                and p.stratum == COLD
                and p.top_k == 5
            ),
        )
        for predicate in predicates:
            matches = sorted(
                (package for package in mode_packages if predicate(package)),
                key=lambda package: package.package_id,
            )
            if not matches:
                raise ValueError(f"cannot construct complete pilot for {mode}")
            chosen.append(matches[0])
    if len({package.package_id for package in chosen}) != len(chosen):
        raise ValueError("pilot package selection contains duplicates")
    return tuple(chosen)


def validate_pilot_adjudication(
    adjudication: PilotAdjudication,
    pilot: tuple[EvidencePackage, ...],
) -> tuple[tuple[EvidencePackage, ...], dict[str, Any]]:
    """Validate and apply the separately frozen pilot ground truth."""

    pilot_ids = tuple(package.package_id for package in pilot)
    if adjudication.pilot_package_ids != pilot_ids:
        raise ValueError("pilot adjudication package IDs or ordering changed")
    pilot_freeze = evidence_freeze(pilot)
    if adjudication.pilot_evidence_signature != pilot_freeze["signature"]:
        raise ValueError("pilot adjudication evidence signature changed")

    retrieved = tuple(
        package for package in pilot if package.condition is EvidenceCondition.RETRIEVED
    )
    retrieved_ids = tuple(package.package_id for package in retrieved)
    record_ids = tuple(record.package_id for record in adjudication.records)
    if record_ids != retrieved_ids:
        raise ValueError("pilot adjudication must cover retrieved packages in pilot order")
    retrieved_freeze = evidence_freeze(retrieved)
    if adjudication.retrieved_pilot_evidence_signature != retrieved_freeze["signature"]:
        raise ValueError("retrieved pilot adjudication evidence signature changed")

    records_by_package = {record.package_id: record for record in adjudication.records}
    packages_by_id = {package.package_id: package for package in retrieved}
    for record in adjudication.records:
        package = packages_by_id[record.package_id]
        if record.target_present != package.target_present:
            raise ValueError(f"adjudication target_present mismatch: {record.package_id}")
        evidence_ids = {item.evidence_id for item in package.evidence}
        if not set(record.supporting_evidence_ids).issubset(evidence_ids):
            raise ValueError(f"adjudication cites unknown evidence: {record.package_id}")

    labeled_pilot = tuple(apply_adjudication(package, records_by_package) for package in pilot)
    unresolved = [
        package.package_id
        for package in labeled_pilot
        if package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
    ]
    if unresolved:
        raise ValueError(f"pilot ground truth remains unresolved: {unresolved}")

    digest = adjudication_digest(adjudication)
    metadata = {
        "schema_version": adjudication.schema_version,
        "adjudication_version": adjudication.adjudication_version,
        "digest_algorithm": "sha256-canonical-json-v1",
        "digest": digest,
        "blinded_to_model_outputs": adjudication.blinded_to_model_outputs,
        "reviewer": adjudication.reviewer,
        "review_date": adjudication.review_date.isoformat(),
        "adjudicated_retrieved_records": len(adjudication.records),
        "retrieved_package_ids": list(record_ids),
        "retrieved_expected_action_counts": dict(
            sorted(Counter(record.expected_action.value for record in adjudication.records).items())
        ),
        "pilot_expected_action_counts": dict(
            sorted(Counter(package.expected_action.value for package in labeled_pilot).items())
        ),
        "borderline_package_ids": [
            record.package_id for record in adjudication.records if record.borderline
        ],
        "pilot_evidence_signature": pilot_freeze["signature"],
        "retrieved_pilot_evidence_signature": retrieved_freeze["signature"],
    }
    return labeled_pilot, metadata


def select_canary(packages: tuple[EvidencePackage, ...]) -> EvidencePackage:
    """Choose one deterministic, answer-expected facts-only oracle record."""

    candidates = sorted(
        (
            package
            for package in packages
            if package.condition is EvidenceCondition.ORACLE_GOLD
            and package.query_mode == "facts_only"
            and package.answer_expected
        ),
        key=lambda package: package.package_id,
    )
    if not candidates:
        raise ValueError("cannot construct an answer-expected facts-only oracle canary")
    return candidates[0]


def _counts(packages: tuple[EvidencePackage, ...]) -> dict[str, Any]:
    return {
        "conditions": dict(sorted(Counter(item.condition.value for item in packages).items())),
        "modes": dict(sorted(Counter(item.query_mode for item in packages).items())),
        "strata": dict(sorted(Counter(item.stratum for item in packages).items())),
        "top_k": dict(
            sorted(
                Counter(
                    "oracle" if item.top_k is None else str(item.top_k) for item in packages
                ).items()
            )
        ),
        "warm_cold": dict(
            sorted(Counter("warm" if item.warm_start else "cold" for item in packages).items())
        ),
        "target_present": dict(
            sorted(Counter(str(item.target_present).lower() for item in packages).items())
        ),
        "evidence_sufficient": dict(
            sorted(
                Counter(
                    "unknown"
                    if item.evidence_sufficient is None
                    else str(item.evidence_sufficient).lower()
                    for item in packages
                ).items()
            )
        ),
        "expected_actions": dict(
            sorted(Counter(item.expected_action.value for item in packages).items())
        ),
    }


def audit_packages(
    packages: tuple[EvidencePackage, ...],
    *,
    evidence_cutoff_year: int,
    test_urls: frozenset[str],
) -> dict[str, Any]:
    oracle = [package for package in packages if package.condition is EvidenceCondition.ORACLE_GOLD]
    retrieved = [
        package for package in packages if package.condition is EvidenceCondition.RETRIEVED
    ]
    oracle_origin_errors = [
        package.package_id
        for package in oracle
        if any(item.origin is not EvidenceOrigin.GOLD_QUERY_ROW for item in package.evidence)
        or any(item.source_url not in test_urls for item in package.evidence)
    ]
    retrieved_leakage = [
        package.package_id
        for package in retrieved
        if any(
            item.origin is not EvidenceOrigin.HISTORICAL_RETRIEVAL
            or item.source_url in test_urls
            or item.source_year > evidence_cutoff_year
            for item in package.evidence
        )
    ]
    if oracle_origin_errors:
        raise ValueError("oracle packages contain evidence outside exact gold test rows")
    if retrieved_leakage:
        raise ValueError("gold or future evidence entered retrieved-context packages")
    unverified_oracle = [
        package.package_id
        for package in oracle
        if not all(item.citation_relationship_verified for item in package.evidence)
    ]
    review_required = [
        package.package_id
        for package in packages
        if package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
    ]
    return {
        "records": len(packages),
        "oracle_gold_records": len(oracle),
        "retrieved_records": len(retrieved),
        "oracle_exact_gold_row_origin_verified": len(oracle) - len(oracle_origin_errors),
        "oracle_citation_relationship_verified": len(oracle) - len(unverified_oracle),
        "oracle_citation_relationship_needs_review": len(unverified_oracle),
        "oracle_needs_review_package_ids": unverified_oracle,
        "retrieved_historical_corpus_verified": len(retrieved) - len(retrieved_leakage),
        "retrieved_gold_or_future_leakage_records": len(retrieved_leakage),
        "target_present_records": sum(package.target_present for package in packages),
        "target_absent_records": sum(not package.target_present for package in packages),
        "expected_action_counts": dict(
            sorted(Counter(package.expected_action.value for package in packages).items())
        ),
        "manual_sufficiency_review_records": len(review_required),
        "manual_sufficiency_review_package_ids": review_required,
    }


def generation_contract(config: RAGConfig) -> dict[str, Any]:
    return {
        "prompt_version": config.settings.prompt_version,
        "prompt_signature": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "output_schema_signature": _canonical_digest(GroundedAnswer.model_json_schema()),
        "model": config.settings.model,
        "reasoning_effort": config.settings.reasoning_effort,
        "verbosity": config.settings.verbosity,
        "max_output_tokens": config.settings.max_output_tokens,
        "temperature": config.settings.temperature,
        "seed": config.settings.seed,
        "top_ks": list(config.top_ks),
        "automatic_retries": config.automatic_retries,
    }


def build_manifest(
    *,
    config: RAGConfig,
    signature: str,
    selected: tuple[Any, ...],
    packages: tuple[EvidencePackage, ...],
    pilot: tuple[EvidencePackage, ...],
    canary: EvidencePackage,
    methodology_audit: dict[str, Any],
    pilot_ground_truth: dict[str, Any],
) -> dict[str, Any]:
    frozen_evidence = evidence_freeze(packages)
    return {
        "schema_version": 4,
        "run_signature": signature,
        "protocol": "bounded_grounded_rag_methodology_v2",
        "methodology_correction": {
            "trigger": "one-call canary exposed case-presence/evidence-sufficiency conflation",
            "oracle_definition": "exact gold citation row from the test query",
            "oracle_label": EvidenceCondition.ORACLE_GOLD.value,
            "retrieved_definition": (
                "unchanged <=2023 historical citation passages -> passage BM25 -> case aggregation"
            ),
            "retrieved_sufficiency_policy": "unknown_needs_review; never inferred from case identity",
        },
        "methodology_audit": methodology_audit,
        "model": config.settings.model,
        "model_revision_policy": "record response.model for every completed request",
        "provider": "OpenAI Responses API",
        "billing": "separate OpenAI API usage; not ChatGPT or Codex allowance",
        "temperature": None,
        "seed": None,
        "reasoning_effort": config.settings.reasoning_effort,
        "automatic_retries": config.automatic_retries,
        "generation_contract": generation_contract(config),
        "selection": {
            "seed": config.seed,
            "queries_per_mode_per_stratum": config.queries_per_stratum,
            "query_modes": list(config.modes),
            "strata": [WARM_SUCCESS, WARM_FAILURE, COLD],
            "underlying_queries": len(selected),
            "records": [
                {
                    "query_id": item.query_id,
                    "mode": item.mode,
                    "stratum": item.stratum,
                }
                for item in selected
            ],
        },
        "generation_plan": {
            "top_ks": list(config.top_ks),
            "planned_logical_requests": len(packages),
            "planned_http_attempts": len(packages),
            "retry_policy": (
                "automatic retries disabled; --retry-errors can add at most one later logical "
                "request for each cached failed record"
            ),
            "counts": _counts(packages),
            "package_ids": [package.package_id for package in packages],
        },
        "pilot": {
            "logical_requests": len(pilot),
            "package_ids": [package.package_id for package in pilot],
            "counts": _counts(pilot),
            "evidence_signature": evidence_freeze(pilot)["signature"],
            "ground_truth": pilot_ground_truth,
        },
        "canary": {
            "logical_requests": 1,
            "selection_policy": "facts-only answer-expected oracle-gold with lowest package ID",
            "package_id": canary.package_id,
            "evidence_signature": package_evidence_lock(canary)["evidence_signature"],
        },
        "evidence_freeze": frozen_evidence,
        "estimate": estimate_tokens_and_cost(packages, config),
        "pilot_estimate": estimate_tokens_and_cost(pilot, config),
        "canary_estimate": estimate_tokens_and_cost((canary,), config),
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def assert_frozen_manifest(frozen: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    """Reject any API execution whose reconstructed protocol or evidence changed."""

    required_keys = (
        "schema_version",
        "run_signature",
        "protocol",
        "model",
        "temperature",
        "seed",
        "reasoning_effort",
        "automatic_retries",
        "generation_contract",
        "selection",
        "generation_plan",
        "pilot",
        "canary",
        "methodology_correction",
        "methodology_audit",
    )
    changed = [key for key in required_keys if frozen.get(key) != rebuilt.get(key)]
    if changed:
        raise ValueError(
            "frozen evaluation protocol mismatch before API generation: " + ", ".join(changed)
        )
    frozen_evidence = frozen.get("evidence_freeze")
    rebuilt_evidence = rebuilt.get("evidence_freeze")
    if not isinstance(frozen_evidence, dict) or "signature" not in frozen_evidence:
        raise ValueError("frozen manifest does not contain an evidence signature")
    if frozen_evidence != rebuilt_evidence:
        raise ValueError("frozen evidence signature mismatch before API generation")


def _frozen_locks(global_manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    evidence = global_manifest.get("evidence_freeze")
    if not isinstance(evidence, dict):
        raise TypeError("global manifest lacks frozen evidence metadata")
    locks = evidence.get("packages")
    if not isinstance(locks, list) or not all(isinstance(lock, dict) for lock in locks):
        raise TypeError("global manifest lacks frozen package locks")
    signature = evidence.get("signature")
    if signature != _canonical_digest(locks):
        raise ValueError("global frozen evidence digest mismatch")
    return tuple(locks)


def _assert_global_manifest_preflight(
    global_manifest: dict[str, Any], config: RAGConfig
) -> tuple[dict[str, Any], ...]:
    if global_manifest.get("schema_version") != 4:
        raise ValueError("global manifest schema version changed")
    if global_manifest.get("generation_contract") != generation_contract(config):
        raise ValueError("frozen generation contract changed")
    if global_manifest.get("model") != config.settings.model:
        raise ValueError("frozen model changed")
    if global_manifest.get("automatic_retries") != 0 or config.automatic_retries != 0:
        raise ValueError("automatic retries must remain disabled")

    locks = _frozen_locks(global_manifest)
    generation_plan = global_manifest.get("generation_plan")
    if not isinstance(generation_plan, dict):
        raise TypeError("global manifest lacks generation plan")
    package_ids = generation_plan.get("package_ids")
    lock_ids = [lock.get("package_id") for lock in locks]
    if package_ids != lock_ids:
        raise ValueError("global frozen package IDs or ordering changed")

    pilot = global_manifest.get("pilot")
    if not isinstance(pilot, dict):
        raise TypeError("global manifest lacks pilot metadata")
    ground_truth = pilot.get("ground_truth")
    if not isinstance(ground_truth, dict) or not isinstance(ground_truth.get("digest"), str):
        raise TypeError("global manifest lacks pilot ground-truth digest")
    expected_signature = _signature_from_evidence_signature(
        config,
        str(global_manifest["evidence_freeze"]["signature"]),
        pilot_ground_truth_digest=ground_truth["digest"],
    )
    if global_manifest.get("run_signature") != expected_signature:
        raise ValueError("global run signature mismatch")
    return locks


def validate_frozen_behaviour_execution(
    *,
    config: RAGConfig,
    global_manifest: dict[str, Any],
    pilot: BehaviourPilotManifest,
    adjudication: BehaviourAdjudication,
    answer_adjudication: PilotAdjudication,
    frozen_packages: FrozenBehaviourPackages,
) -> tuple[tuple[EvidencePackage, ...], tuple[EvidencePackage, ...]]:
    """Verify the input-only pilot artifact without rebuilding retrieval."""

    locks = _assert_global_manifest_preflight(global_manifest, config)
    if pilot.global_manifest_schema_version != global_manifest["schema_version"]:
        raise ValueError("behaviour pilot global manifest schema changed")
    if pilot.global_run_signature != global_manifest["run_signature"]:
        raise ValueError("behaviour pilot global run signature changed")
    if pilot.global_evidence_digest != global_manifest["evidence_freeze"]["signature"]:
        raise ValueError("behaviour pilot global evidence digest changed")
    if pilot.generation_contract != generation_contract(config):
        raise ValueError("behaviour pilot generation contract changed")
    if pilot.pilot_version != adjudication.pilot_version:
        raise ValueError("behaviour pilot version mismatch")
    if pilot.adjudication_version != adjudication.adjudication_version:
        raise ValueError("behaviour adjudication version mismatch")

    adjudication_digest_value = behaviour_adjudication_digest(adjudication)
    if pilot.adjudication_digest != adjudication_digest_value:
        raise ValueError("behaviour adjudication digest mismatch")
    if adjudication_digest(answer_adjudication) != pilot.answer_pilot_adjudication_digest:
        raise ValueError("answer-only pilot adjudication changed")
    if answer_adjudication.adjudication_version != pilot.answer_pilot_adjudication_version:
        raise ValueError("answer-only pilot adjudication version changed")
    if answer_adjudication.pilot_evidence_signature != pilot.answer_pilot_evidence_digest:
        raise ValueError("answer-only pilot evidence changed")

    if frozen_packages.pilot_version != pilot.pilot_version:
        raise ValueError("frozen package pilot version mismatch")
    if frozen_packages.run_signature != pilot.run_signature:
        raise ValueError("frozen package run signature mismatch")
    selected_ids = pilot.selected_package_ids
    if adjudication.selected_package_ids != selected_ids:
        raise ValueError("behaviour adjudication package IDs or ordering changed")
    if frozen_packages.selected_package_ids != selected_ids:
        raise ValueError("frozen execution package IDs or ordering changed")
    if len(frozen_packages.packages) != 12:
        raise ValueError("frozen behaviour execution requires exactly 12 packages")
    if tuple(package.package_id for package in frozen_packages.packages) != selected_ids:
        raise ValueError("frozen execution package IDs or ordering changed")

    locks_by_id = {str(lock.get("package_id")): lock for lock in locks}
    try:
        selected_locks = [locks_by_id[package_id] for package_id in selected_ids]
    except KeyError as error:
        raise ValueError(
            f"frozen execution package is absent from global locks: {error}"
        ) from error
    if canonical_digest(selected_locks) != pilot.evidence_digest:
        raise ValueError("behaviour pilot evidence digest mismatch")
    for package, expected_lock in zip(frozen_packages.packages, selected_locks, strict=True):
        if package_evidence_lock(package) != expected_lock:
            raise ValueError(f"frozen package evidence or input changed: {package.package_id}")

    reviews_by_id = {
        record.package_id: record
        for record in adjudication.reviewed_candidates + adjudication.oracle_reviews
    }
    raw_packages = frozen_packages.packages
    for package in raw_packages:
        record = reviews_by_id.get(package.package_id)
        if record is None:
            raise ValueError(f"frozen package lacks adjudication: {package.package_id}")
        if (
            package.query_mode != record.query_mode
            or package.condition is not record.condition
            or package.top_k != record.top_k
            or package.target_present != record.target_present
        ):
            raise ValueError(f"review metadata changed: {package.package_id}")
        evidence_ids = {item.evidence_id for item in package.evidence}
        if not set(record.supporting_evidence_ids).issubset(evidence_ids):
            raise ValueError(f"review cites unknown evidence: {package.package_id}")
    labeled_packages = tuple(
        apply_behaviour_labels(package, reviews_by_id) for package in raw_packages
    )
    counts = {
        "expected_actions": dict(
            sorted(Counter(package.expected_action.value for package in labeled_packages).items())
        ),
        "query_modes": dict(
            sorted(Counter(package.query_mode for package in labeled_packages).items())
        ),
        "conditions": dict(
            sorted(Counter(package.condition.value for package in labeled_packages).items())
        ),
        "top_k": dict(
            sorted(
                Counter(
                    "oracle" if package.top_k is None else str(package.top_k)
                    for package in labeled_packages
                ).items()
            )
        ),
    }
    if counts != pilot.counts:
        raise ValueError("behaviour pilot counts changed")

    expected_run_signature = behaviour_run_signature(
        global_run_signature=pilot.global_run_signature,
        adjudication_digest_value=adjudication_digest_value,
        evidence_digest=pilot.evidence_digest,
    )
    if pilot.run_signature != expected_run_signature:
        raise ValueError("behaviour pilot run signature changed")
    return raw_packages, labeled_packages


def preflight_frozen_behaviour_execution(
    *,
    rag_config_path: Path,
    global_manifest_path: Path,
    behaviour_manifest_path: Path,
    behaviour_packages_path: Path,
    behaviour_adjudication_path: Path,
    answer_adjudication_path: Path,
) -> tuple[
    RAGConfig,
    BehaviourPilotManifest,
    tuple[EvidencePackage, ...],
    tuple[EvidencePackage, ...],
]:
    config = load_config(rag_config_path)
    global_manifest = load_json(global_manifest_path)
    pilot = load_behaviour_pilot(behaviour_manifest_path)
    frozen_packages = load_frozen_behaviour_packages(behaviour_packages_path)
    adjudication = load_behaviour_adjudication(behaviour_adjudication_path)
    answer_adjudication = load_pilot_adjudication(answer_adjudication_path)
    raw_packages, labeled_packages = validate_frozen_behaviour_execution(
        config=config,
        global_manifest=global_manifest,
        pilot=pilot,
        adjudication=adjudication,
        answer_adjudication=answer_adjudication,
        frozen_packages=frozen_packages,
    )
    return config, pilot, raw_packages, labeled_packages


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manual_review_template(
    records: list[GenerationRecord], *, count: int, seed: int
) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[GenerationRecord]] = defaultdict(list)
    for record in records:
        buckets[(record.package.query_mode, record.package.condition.value)].append(record)
    for key, items in buckets.items():
        items.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}\0{key}\0{record.package.package_id}".encode()
            ).hexdigest()
        )
    selected: list[GenerationRecord] = []
    keys = sorted(buckets)
    while len(selected) < min(count, len(records)) and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    return {
        "schema_version": 2,
        "instructions": (
            "Inspect query, evidence, and answer. Fill booleans without using outside legal "
            "knowledge. Precedent relevance asks whether the passage supports identifying the "
            "recommended authority, not whether the client ultimately satisfies its test. Review "
            "the supported proposition, any necessary factual-application limitation, and any "
            "unsupported factual conclusion separately."
        ),
        "records": [
            {
                "package_id": record.package.package_id,
                "mode": record.package.query_mode,
                "condition": record.package.condition.value,
                "warm_start": record.package.warm_start,
                "top_k": record.package.top_k,
                "query": record.package.query_text,
                "evidence": [item.model_dump(mode="json") for item in record.package.evidence],
                "answer": (
                    record.result.answer.model_dump(mode="json")
                    if record.result.answer is not None
                    else None
                ),
                "automated_issues": [
                    item.model_dump(mode="json")
                    for item in evaluate_record(record).validation.issues
                ],
                "review": {
                    "precedent_relevance_supported": None,
                    "semantic_entailment": None,
                    "factual_application_limit_appropriate": None,
                    "unsupported_factual_conclusion_present": None,
                    "citation_complete": None,
                    "unsupported_claim_present": None,
                    "abstention_appropriate": None,
                    "notes": "",
                },
            }
            for record in selected
        ],
    }


def sufficiency_review_template(packages: tuple[EvidencePackage, ...]) -> dict[str, Any]:
    review = sorted(
        (
            package
            for package in packages
            if package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
        ),
        key=lambda package: package.package_id,
    )
    return {
        "schema_version": 1,
        "instructions": (
            "Decide only whether the supplied evidence supports a defensible precedent answer to "
            "the query. Case identity alone is not sufficient. Set evidence_sufficient and "
            "expected_action to answer or abstain; use notes for ambiguity."
        ),
        "records": [
            {
                "package_id": package.package_id,
                "query_id": package.query_id,
                "mode": package.query_mode,
                "condition": package.condition.value,
                "stratum": package.stratum,
                "top_k": package.top_k,
                "target_present": package.target_present,
                "query": package.query_text,
                "accepted_case_ids": list(package.accepted_case_ids),
                "evidence": [item.model_dump(mode="json") for item in package.evidence],
                "review": {
                    "evidence_sufficient": None,
                    "expected_action": None,
                    "notes": "",
                },
            }
            for package in review
        ],
    }


def all_generation_attempts_failed(
    attempted_records: list[GenerationRecord], records: list[GenerationRecord]
) -> bool:
    """Fail fresh all-error runs and cached all-error resumptions."""

    results = attempted_records if attempted_records else records
    return bool(results) and all(
        record.result.call_status is not ProviderCallStatus.SUCCEEDED for record in results
    )


def assert_cached_record_matches_execution(
    record: GenerationRecord,
    *,
    package: EvidencePackage,
    settings: GenerationSettings,
    run_signature: str,
) -> None:
    if record.run_signature != run_signature:
        raise ValueError("cached generation run signature changed")
    if record.package != package:
        raise ValueError(f"cached generation package changed: {package.package_id}")
    if record.prompt_version != settings.prompt_version:
        raise ValueError(f"cached prompt version changed: {package.package_id}")
    if record.system_instructions != SYSTEM_INSTRUCTIONS:
        raise ValueError(f"cached system instructions changed: {package.package_id}")
    if record.user_input != render_user_input(package):
        raise ValueError(f"cached rendered input changed: {package.package_id}")
    if record.settings != settings:
        raise ValueError(f"cached generation settings changed: {package.package_id}")


def prepare(
    *, data_dir: Path, splits: Path, corpus_config_path: Path, rag_config_path: Path
) -> tuple[
    RAGConfig,
    tuple[Any, ...],
    tuple[EvidencePackage, ...],
    dict[str, Any],
]:
    corpus_config = load_corpus_config(corpus_config_path)
    rag_config = load_config(rag_config_path)
    dataset = load_corpus_repair_dataset(
        data_dir / "COMBINED_ALL_CASES_FINAL_V2.csv",
        splits,
        evidence_cutoff_year=corpus_config.evidence_cutoff_year,
        max_passage_chars=corpus_config.max_passage_chars,
        max_profile_passages=corpus_config.max_profile_passages,
        max_profile_identifier_chars=corpus_config.max_profile_identifier_chars,
        max_profile_context_chars=corpus_config.max_profile_context_chars,
        max_profile_chars=corpus_config.max_profile_chars,
    )
    index = BM25Index(
        tuple(context.text for context in dataset.contexts),
        k1=corpus_config.k1,
        b=corpus_config.b,
    )
    case_to_id = {key: index for index, key in enumerate(dataset.case_keys)}
    selected = select_queries(
        modes=rag_config.modes,
        dataset=dataset,
        index=index,
        case_to_id=case_to_id,
        per_stratum=rag_config.queries_per_stratum,
        retrieval_depth=max(rag_config.top_ks),
        seed=rag_config.seed,
    )
    packages = build_packages(
        selected,
        top_ks=rag_config.top_ks,
        index=index,
        dataset=dataset,
        case_to_id=case_to_id,
    )
    methodology_audit = audit_packages(
        packages,
        evidence_cutoff_year=corpus_config.evidence_cutoff_year,
        test_urls=dataset.test_urls,
    )
    return (
        rag_config,
        selected,
        packages,
        methodology_audit,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute bounded grounded RAG evaluation"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--corpus-config", type=Path, default=DEFAULT_CORPUS_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-adjudication", type=Path, default=DEFAULT_PILOT_ADJUDICATION)
    parser.add_argument(
        "--behaviour-adjudication",
        type=Path,
        default=DEFAULT_BEHAVIOUR_ADJUDICATION,
    )
    parser.add_argument("--behaviour-manifest", type=Path, default=DEFAULT_BEHAVIOUR_PILOT)
    parser.add_argument("--behaviour-packages", type=Path, default=DEFAULT_BEHAVIOUR_PACKAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument("--sufficiency-review", type=Path, default=DEFAULT_SUFFICIENCY_REVIEW)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute",
        action="store_true",
        help="make usage-billed API requests; omit for the safe offline preparation default",
    )
    execution.add_argument(
        "--reconstruct-and-verify",
        action="store_true",
        help="rebuild retrieval and compare it with frozen artifacts without provider access",
    )
    execution.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify frozen behavioural inputs and stop before provider construction",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--pilot", action="store_true", help="with --execute, run only the fixed 12-record pilot"
    )
    scope.add_argument(
        "--canary",
        action="store_true",
        help="with --execute, run only the frozen one-record oracle canary",
    )
    scope.add_argument(
        "--behaviour-pilot",
        action="store_true",
        help="with --execute, run only the frozen 12-record balanced behavioural pilot",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --execute, retry each cached failed record once in this invocation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preflight_only and not args.behaviour_pilot:
            raise ValueError("--preflight-only requires --behaviour-pilot")

        direct_behaviour = args.behaviour_pilot and (args.execute or args.preflight_only)
        frozen_manifest: dict[str, Any] | None
        behaviour_manifest: BehaviourPilotManifest | None = None
        behaviour_targets: tuple[EvidencePackage, ...] = ()
        labeled_behaviour_targets: tuple[EvidencePackage, ...] = ()
        if direct_behaviour:
            frozen_manifest = load_json(args.manifest)
            (
                config,
                behaviour_manifest,
                behaviour_targets,
                labeled_behaviour_targets,
            ) = preflight_frozen_behaviour_execution(
                rag_config_path=args.config,
                global_manifest_path=args.manifest,
                behaviour_manifest_path=args.behaviour_manifest,
                behaviour_packages_path=args.behaviour_packages,
                behaviour_adjudication_path=args.behaviour_adjudication,
                answer_adjudication_path=args.pilot_adjudication,
            )
            manifest = frozen_manifest
            packages = behaviour_targets
            pilot: tuple[EvidencePackage, ...] = ()
            labeled_pilot: tuple[EvidencePackage, ...] = ()
            adjudication = load_pilot_adjudication(args.pilot_adjudication)
            pilot_ground_truth: dict[str, Any] = {}
            signature = behaviour_manifest.global_run_signature
            canary: EvidencePackage | None = None
            print(
                "verified 12 frozen behavioural packages, protocol, adjudication, and run "
                f"signature: {behaviour_manifest.run_signature}"
            )
            if args.preflight_only:
                print("frozen behavioural preflight complete; provider was not constructed")
                return 0
        else:
            frozen_manifest = (
                load_json(args.manifest) if args.execute or args.reconstruct_and_verify else None
            )
            config, selected, packages, methodology_audit = prepare(
                data_dir=args.data_dir,
                splits=args.splits,
                corpus_config_path=args.corpus_config,
                rag_config_path=args.config,
            )
            pilot = select_pilot(packages)
            adjudication = load_pilot_adjudication(args.pilot_adjudication)
            labeled_pilot, pilot_ground_truth = validate_pilot_adjudication(adjudication, pilot)
            signature = _signature(
                config,
                packages,
                pilot_ground_truth_digest=pilot_ground_truth["digest"],
            )
            canary = select_canary(packages)
            manifest = build_manifest(
                config=config,
                signature=signature,
                selected=selected,
                packages=packages,
                pilot=pilot,
                canary=canary,
                methodology_audit=methodology_audit,
                pilot_ground_truth=pilot_ground_truth,
            )
            if args.behaviour_pilot:
                behaviour_adjudication = load_behaviour_adjudication(args.behaviour_adjudication)
                behaviour_manifest = load_behaviour_pilot(args.behaviour_manifest)
                behaviour_targets, labeled_behaviour_targets = validate_behaviour_pilot(
                    adjudication=behaviour_adjudication,
                    pilot=behaviour_manifest,
                    packages=packages,
                    global_manifest=(frozen_manifest if frozen_manifest is not None else manifest),
                    answer_adjudication=adjudication,
                )
            print(
                f"planned requests: {len(packages)}; pilot: {len(pilot)}; "
                f"canary: 1; run signature: {signature}"
            )
            if args.reconstruct_and_verify:
                assert frozen_manifest is not None
                assert_frozen_manifest(frozen_manifest, manifest)
                if args.behaviour_pilot:
                    assert behaviour_manifest is not None
                    frozen_packages = load_frozen_behaviour_packages(args.behaviour_packages)
                    direct_raw, _ = validate_frozen_behaviour_execution(
                        config=config,
                        global_manifest=frozen_manifest,
                        pilot=behaviour_manifest,
                        adjudication=behaviour_adjudication,
                        answer_adjudication=adjudication,
                        frozen_packages=frozen_packages,
                    )
                    if direct_raw != behaviour_targets:
                        raise ValueError(
                            "frozen behaviour packages differ from reconstructed packages"
                        )
                print("full reconstruction matches frozen artifacts; no API requests made")
                return 0
            if not args.execute:
                write_json(args.manifest, manifest)
                write_json(args.sufficiency_review, sufficiency_review_template(packages))
                print(f"wrote offline manifest {args.manifest}")
                print(f"wrote private sufficiency-review queue {args.sufficiency_review}")
                print("offline preparation complete; no API requests made")
                return 0

            assert frozen_manifest is not None
            assert_frozen_manifest(frozen_manifest, manifest)
            print(
                "verified frozen protocol and evidence signature before API generation: "
                f"{manifest['evidence_freeze']['signature']}"
            )

        targets = (
            behaviour_targets
            if args.behaviour_pilot
            else (canary,)
            if args.canary and canary is not None
            else pilot
            if args.pilot
            else packages
        )
        execution_signature = (
            behaviour_manifest.run_signature if behaviour_manifest is not None else signature
        )
        generator: OpenAIResponsesGenerator | None = None
        records: list[GenerationRecord] = []
        attempted_records: list[GenerationRecord] = []
        for position, package in enumerate(targets, start=1):
            path = cache_path(args.cache_dir, execution_signature, package.package_id)
            cached = load_record(
                path,
                run_signature=execution_signature,
                package_id=package.package_id,
            )
            if cached is not None and not (args.retry_errors and cached.result.error is not None):
                assert_cached_record_matches_execution(
                    cached,
                    package=package,
                    settings=config.settings,
                    run_signature=execution_signature,
                )
                record = cached
            else:
                if generator is None:
                    generator = OpenAIResponsesGenerator()
                record = generate_record(
                    generator,
                    package,
                    config.settings,
                    execution_signature,
                )
                save_record(path, record)
                attempted_records.append(record)
            records.append(record)
            print(f"generation {position}/{len(targets)}: {package.package_id}", flush=True)

        if args.behaviour_pilot:
            labels_by_id = {package.package_id: package for package in labeled_behaviour_targets}
            evaluation_records = [
                record.model_copy(update={"package": labels_by_id[record.package.package_id]})
                for record in records
            ]
        else:
            records_by_package = {record.package_id: record for record in adjudication.records}
            evaluation_records = [
                record.model_copy(
                    update={"package": apply_adjudication(record.package, records_by_package)}
                )
                for record in records
            ]
        if args.pilot and tuple(record.package for record in evaluation_records) != labeled_pilot:
            raise ValueError("generated pilot does not match frozen adjudicated pilot")

        result = {
            "schema_version": 2,
            "run_signature": execution_signature,
            "evidence_signature": manifest["evidence_freeze"]["signature"],
            "pilot_ground_truth": (
                {
                    "pilot_version": behaviour_manifest.pilot_version,
                    "adjudication_version": behaviour_manifest.adjudication_version,
                    "adjudication_digest": behaviour_manifest.adjudication_digest,
                    "evidence_digest": behaviour_manifest.evidence_digest,
                }
                if behaviour_manifest is not None
                else pilot_ground_truth
            ),
            "scope": (
                "behaviour_pilot"
                if args.behaviour_pilot
                else "canary"
                if args.canary
                else "pilot"
                if args.pilot
                else "full"
            ),
            "model": config.settings.model,
            "aggregates": grouped_summaries(evaluation_records),
            "outcomes": [
                evaluate_record(record).model_dump(mode="json") for record in evaluation_records
            ],
            "manual_semantic_review": "pending; template written outside version control",
        }
        suffix = (
            "_behaviour_pilot"
            if args.behaviour_pilot
            else "_canary"
            if args.canary
            else "_pilot"
            if args.pilot
            else ""
        )
        output = args.output.with_name(f"{args.output.stem}{suffix}{args.output.suffix}")
        write_json(output, result)
        write_json(
            args.manual_review,
            manual_review_template(
                evaluation_records,
                count=config.manual_review_records,
                seed=config.seed,
            ),
        )
        print(f"wrote {output}")
        print(f"wrote private manual-review template {args.manual_review}")
        if all_generation_attempts_failed(attempted_records, records):
            print("RAG generation failed: all provider attempts failed", file=sys.stderr)
            return 1
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"RAG evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
