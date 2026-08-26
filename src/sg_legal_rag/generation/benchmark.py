from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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

from .evaluation import evaluate_record, grouped_summaries
from .evidence import EvidenceCondition, EvidencePackage
from .provider import (
    SYSTEM_INSTRUCTIONS,
    GenerationRecord,
    GenerationSettings,
    OpenAIResponsesGenerator,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "rag_baseline.json"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "processed" / "generation"
DEFAULT_MANUAL_REVIEW = PROJECT_ROOT / "data" / "processed" / "rag_manual_review.json"
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


def _signature(config: RAGConfig, packages: tuple[EvidencePackage, ...]) -> str:
    payload = {
        "cache_schema": 1,
        "config": asdict(config),
        "packages": [
            {
                "package_id": package.package_id,
                "query_id": package.query_id,
                "mode": package.query_mode,
                "condition": package.condition.value,
                "top_k": package.top_k,
                "evidence": [
                    [item.case_id, item.passage_digest, item.retrieval_score]
                    for item in package.evidence
                ],
            }
            for package in packages
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


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
    """Two-mode, 12-call pilot spanning oracle, retrieval, abstention, and k endpoints."""

    chosen: list[EvidencePackage] = []
    modes = sorted({package.query_mode for package in packages})
    for mode in modes:
        mode_packages = [package for package in packages if package.query_mode == mode]
        predicates = (
            lambda p: p.condition is EvidenceCondition.ORACLE and p.stratum == WARM_SUCCESS,
            lambda p: p.condition is EvidenceCondition.RETRIEVED and p.top_k == 1,
            lambda p: p.condition is EvidenceCondition.RETRIEVED and p.top_k == 5,
            lambda p: (
                p.condition is EvidenceCondition.INSUFFICIENT
                and p.stratum == WARM_FAILURE
                and p.top_k == 5
            ),
            lambda p: (
                p.condition is EvidenceCondition.INSUFFICIENT and p.stratum == COLD and p.top_k == 1
            ),
            lambda p: (
                p.condition is EvidenceCondition.INSUFFICIENT and p.stratum == COLD and p.top_k == 5
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
    }


def build_manifest(
    *,
    config: RAGConfig,
    signature: str,
    selected: tuple[Any, ...],
    packages: tuple[EvidencePackage, ...],
    pilot: tuple[EvidencePackage, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_signature": signature,
        "protocol": "bounded_grounded_rag",
        "model": config.settings.model,
        "model_revision_policy": "record response.model for every completed request",
        "provider": "OpenAI Responses API",
        "billing": "separate OpenAI API usage; not ChatGPT or Codex allowance",
        "temperature": None,
        "seed": None,
        "reasoning_effort": config.settings.reasoning_effort,
        "automatic_retries": config.automatic_retries,
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
        },
        "estimate": estimate_tokens_and_cost(packages, config),
        "pilot_estimate": estimate_tokens_and_cost(pilot, config),
    }


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
        "schema_version": 1,
        "instructions": (
            "Inspect query, evidence, and answer. Fill booleans without using outside legal "
            "knowledge; semantic_entailment asks only whether the passage supports each claim."
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
                    "semantic_entailment": None,
                    "citation_complete": None,
                    "unsupported_claim_present": None,
                    "abstention_appropriate": None,
                    "notes": "",
                },
            }
            for record in selected
        ],
    }


def prepare(
    *, data_dir: Path, splits: Path, corpus_config_path: Path, rag_config_path: Path
) -> tuple[RAGConfig, tuple[Any, ...], tuple[EvidencePackage, ...], str]:
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
    return rag_config, selected, packages, _signature(rag_config, packages)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute bounded grounded RAG evaluation"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--corpus-config", type=Path, default=DEFAULT_CORPUS_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make usage-billed API requests; omit for the safe offline preparation default",
    )
    parser.add_argument(
        "--pilot", action="store_true", help="with --execute, run only the fixed 12-record pilot"
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
        config, selected, packages, signature = prepare(
            data_dir=args.data_dir,
            splits=args.splits,
            corpus_config_path=args.corpus_config,
            rag_config_path=args.config,
        )
        pilot = select_pilot(packages)
        manifest = build_manifest(
            config=config,
            signature=signature,
            selected=selected,
            packages=packages,
            pilot=pilot,
        )
        write_json(args.manifest, manifest)
        print(f"wrote offline manifest {args.manifest}")
        print(f"planned requests: {len(packages)}; pilot: {len(pilot)}; run signature: {signature}")
        if not args.execute:
            print("offline preparation complete; no API requests made")
            return 0

        targets = pilot if args.pilot else packages
        generator = OpenAIResponsesGenerator()
        records: list[GenerationRecord] = []
        for position, package in enumerate(targets, start=1):
            path = cache_path(args.cache_dir, signature, package.package_id)
            cached = load_record(path, run_signature=signature, package_id=package.package_id)
            if cached is not None and not (args.retry_errors and cached.result.error is not None):
                record = cached
            else:
                record = generate_record(generator, package, config.settings, signature)
                save_record(path, record)
            records.append(record)
            print(f"generation {position}/{len(targets)}: {package.package_id}", flush=True)

        result = {
            "schema_version": 1,
            "run_signature": signature,
            "scope": "pilot" if args.pilot else "full",
            "model": config.settings.model,
            "aggregates": grouped_summaries(records),
            "outcomes": [evaluate_record(record).model_dump(mode="json") for record in records],
            "manual_semantic_review": "pending; template written outside version control",
        }
        output = args.output.with_name(
            f"{args.output.stem}_pilot{args.output.suffix}" if args.pilot else args.output.name
        )
        write_json(output, result)
        write_json(
            args.manual_review,
            manual_review_template(records, count=config.manual_review_records, seed=config.seed),
        )
        print(f"wrote {output}")
        print(f"wrote private manual-review template {args.manual_review}")
        return 0
    except (KeyError, OSError, ValueError) as error:
        print(f"RAG evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
