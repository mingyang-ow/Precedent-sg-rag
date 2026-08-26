from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from sg_legal_rag.generation.benchmark import (
    RAGConfig,
    all_generation_attempts_failed,
    assert_frozen_manifest,
    audit_packages,
    build_manifest,
    evidence_freeze,
    select_canary,
    select_pilot,
    sufficiency_review_template,
)
from sg_legal_rag.generation.evaluation import (
    EvaluationStatus,
    evaluate_record,
    summarize_records,
    validate_answer,
)
from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidenceOrigin,
    EvidencePackage,
    EvidenceSufficiencyBasis,
    ExpectedAction,
    package_oracle_evidence,
    package_retrieved_evidence,
    prompt_evidence,
    retrieve_passages,
)
from sg_legal_rag.generation.provider import (
    SYSTEM_INSTRUCTIONS,
    GenerationRecord,
    GenerationSettings,
    OpenAIResponsesGenerator,
    ProviderCallStatus,
    ProviderResult,
    TokenUsage,
    cache_path,
    generate_record,
    load_record,
    render_user_input,
    save_record,
)
from sg_legal_rag.generation.sampling import (
    COLD,
    WARM_FAILURE,
    WARM_SUCCESS,
    build_packages,
    select_queries,
)
from sg_legal_rag.generation.schema import AnswerStatus, GroundedAnswer, GroundedClaim
from sg_legal_rag.retrieval.benchmark import GoldCitationContext, QueryRecord
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset, HistoricalContext


def context(case_key: str, raw_case: str, reference: str, text: str, digest: str):
    return HistoricalContext(
        case_key=case_key,
        raw_case=raw_case,
        source_url=f"https://example.test/{digest}",
        source_reference=reference,
        source_year=2023,
        text=text,
        original_chars=len(text),
        identifier_matched=True,
        digest=digest,
    )


def gold_query(
    *,
    text: str,
    case_key: str,
    raw_case: str,
    row_id: str,
    principle: str = "test principle",
    combined: bool = False,
) -> QueryRecord:
    fact = text.removesuffix(f" {principle}") if combined else text
    paragraph = f"The test judgment applies {raw_case} to {principle}."
    gold = GoldCitationContext(
        row_id=row_id,
        case_key=case_key,
        raw_case=raw_case,
        source_url=f"https://test.example/{row_id}",
        source_reference="[2024] SGHC 1",
        source_year=2024,
        fact_query=fact,
        principle=principle,
        paragraph=paragraph,
        paragraph_digest=hashlib.sha256(paragraph.encode()).hexdigest(),
        identifier_matched=True,
    )
    return QueryRecord(text, {case_key}, gold_contexts={row_id: gold})


@pytest.fixture
def corpus() -> CorpusRepairDataset:
    alpha = "Alpha v Beta [2020] SGCA 2"
    beta = "Beta v Crown [2019] SGHC 4"
    cold = "Cold v Start [2023] SGHC 9"
    contexts = (
        context(
            alpha.casefold(),
            alpha,
            "[2023] SGCA 1",
            "The court applied Alpha v Beta to objective contract interpretation.",
            "alpha-digest",
        ),
        context(
            beta.casefold(),
            beta,
            "[2022] SGHC 8",
            "Beta v Crown concerned sentencing proportionality in a criminal appeal.",
            "beta-digest",
        ),
    )
    queries = {
        "facts_only": [
            gold_query(
                text="objective contract interpretation",
                case_key=alpha.casefold(),
                raw_case=alpha,
                row_id="facts-alpha",
            ),
            gold_query(
                text="objective contract unmatched terminology",
                case_key=beta.casefold(),
                raw_case=beta,
                row_id="facts-beta",
            ),
            gold_query(
                text="objective cold-start duty",
                case_key=cold.casefold(),
                raw_case=cold,
                row_id="facts-cold",
            ),
        ],
        "principle_only": [],
        "facts_principle": [
            gold_query(
                text="contract facts objective interpretation",
                case_key=alpha.casefold(),
                raw_case=alpha,
                row_id="combined-alpha",
                principle="objective interpretation",
                combined=True,
            ),
            gold_query(
                text="contract absent principle",
                case_key=beta.casefold(),
                raw_case=beta,
                row_id="combined-beta",
                principle="principle",
                combined=True,
            ),
            gold_query(
                text="sentencing new cold principle",
                case_key=cold.casefold(),
                raw_case=cold,
                row_id="combined-cold",
                principle="principle",
                combined=True,
            ),
        ],
    }
    return CorpusRepairDataset(
        case_keys=(alpha.casefold(), beta.casefold(), cold.casefold()),
        case_texts=(alpha, beta, cold),
        contexts=contexts,
        context_case_ids=np.asarray([0, 1], dtype=np.int64),
        profiles=(alpha, beta, cold),
        historical_case_ids=frozenset({0, 1}),
        queries_by_mode=queries,
        audit={},
        test_urls=frozenset(
            {
                "https://test.example/facts-alpha",
                "https://test.example/facts-beta",
                "https://test.example/facts-cold",
                "https://test.example/combined-alpha",
                "https://test.example/combined-beta",
                "https://test.example/combined-cold",
            }
        ),
        max_passage_chars=200,
    )


@pytest.fixture
def index(corpus: CorpusRepairDataset) -> BM25Index:
    return BM25Index([item.text for item in corpus.contexts])


@pytest.fixture
def case_to_id(corpus: CorpusRepairDataset) -> dict[str, int]:
    return {key: position for position, key in enumerate(corpus.case_keys)}


@pytest.fixture
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


def retrieved_package(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    query_position: int,
    *,
    top_k: int = 1,
) -> EvidencePackage:
    query = corpus.queries_by_mode["facts_only"][query_position]
    stratum = (WARM_SUCCESS, WARM_FAILURE, COLD)[query_position]
    return package_retrieved_evidence(
        mode="facts_only",
        query=query,
        stratum=stratum,
        top_k=top_k,
        index=index,
        dataset=corpus,
        case_to_id=case_to_id,
    )


def reviewed_package(package: EvidencePackage, *, sufficient: bool) -> EvidencePackage:
    payload = package.model_dump(mode="python")
    payload.update(
        evidence_sufficient=sufficient,
        expected_action=ExpectedAction.ANSWER if sufficient else ExpectedAction.ABSTAIN,
        sufficiency_basis=(
            EvidenceSufficiencyBasis.MANUAL_REVIEWED_SUFFICIENT
            if sufficient
            else EvidenceSufficiencyBasis.MANUAL_REVIEWED_INSUFFICIENT
        ),
    )
    return EvidencePackage.model_validate(payload)


def answer_for(package: EvidencePackage, *, recommendation: str | None = None) -> GroundedAnswer:
    evidence = package.evidence[0]
    quote = " ".join(evidence.passage.split()[:5])
    return GroundedAnswer(
        status=AnswerStatus.ANSWERED,
        recommended_case_id=recommendation or evidence.case_id,
        explanation="The supplied passage supports the recommendation.",
        claims=[
            GroundedClaim(
                statement="The precedent addresses the query.",
                evidence_id=evidence.evidence_id,
                supporting_quote=quote,
            )
        ],
    )


def record_for(
    package: EvidencePackage,
    answer: GroundedAnswer | None,
    settings: GenerationSettings,
    *,
    error: str | None = None,
    response_id: str | None = "resp_test",
) -> GenerationRecord:
    return GenerationRecord(
        run_signature="signature",
        package=package,
        prompt_version=settings.prompt_version,
        system_instructions=SYSTEM_INSTRUCTIONS,
        user_input=render_user_input(package),
        settings=settings,
        result=ProviderResult(
            requested_model=settings.model,
            returned_model=settings.model,
            response_id=response_id,
            generated_at="2026-08-26T00:00:00+00:00",
            latency_ms=10,
            usage=TokenUsage(
                input_tokens=100,
                cached_input_tokens=0,
                output_tokens=20,
                reasoning_tokens=0,
                total_tokens=120,
            ),
            estimated_cost_usd=0.000044,
            raw_output="{}",
            answer=answer,
            error=error,
        ),
    )


def test_retrieval_packages_best_positive_passage_per_case(
    corpus: CorpusRepairDataset, index: BM25Index
) -> None:
    evidence = retrieve_passages(index, corpus, "objective contract", top_k=5)

    assert [item.case_id for item in evidence] == ["case:0"]
    assert evidence[0].evidence_id == "E1"
    assert evidence[0].retrieval_rank == 1
    assert evidence[0].retrieval_score is not None
    assert evidence[0].passage_digest == "alpha-digest"


def test_oracle_context_uses_exact_gold_query_row_not_historical_evidence(
    corpus: CorpusRepairDataset, case_to_id: dict[str, int]
) -> None:
    package = package_oracle_evidence(
        mode="facts_only",
        query=corpus.queries_by_mode["facts_only"][0],
        stratum=WARM_SUCCESS,
        dataset=corpus,
        case_to_id=case_to_id,
    )

    assert package.condition is EvidenceCondition.ORACLE_GOLD
    assert package.top_k is None
    assert package.evidence[0].case_id == "case:0"
    assert package.evidence[0].origin is EvidenceOrigin.GOLD_QUERY_ROW
    assert package.evidence[0].gold_row_id == "facts-alpha"
    assert package.evidence[0].source_year == 2024
    assert (
        package.evidence[0].passage_digest
        == hashlib.sha256(package.evidence[0].passage.encode("utf-8")).hexdigest()
    )
    assert package.target_present
    assert package.evidence_sufficient is True
    assert package.expected_action is ExpectedAction.ANSWER


def test_oracle_gold_context_is_available_for_cold_query(
    corpus: CorpusRepairDataset, case_to_id: dict[str, int]
) -> None:
    package = package_oracle_evidence(
        mode="facts_only",
        query=corpus.queries_by_mode["facts_only"][2],
        stratum=COLD,
        dataset=corpus,
        case_to_id=case_to_id,
    )

    assert package.condition is EvidenceCondition.ORACLE_GOLD
    assert package.evidence[0].case_id == "case:2"
    assert package.evidence[0].gold_row_id == "facts-cold"
    assert not package.warm_start
    assert package.expected_action is ExpectedAction.ANSWER


def test_gold_or_future_evidence_never_enters_retrieved_context(
    corpus: CorpusRepairDataset, index: BM25Index
) -> None:
    evidence = retrieve_passages(index, corpus, "objective contract", top_k=5)

    assert evidence
    assert all(item.origin is EvidenceOrigin.HISTORICAL_RETRIEVAL for item in evidence)
    assert all(item.gold_row_id is None for item in evidence)
    assert all(item.source_year <= 2023 for item in evidence)
    assert all(item.source_url not in corpus.test_urls for item in evidence)


def test_prompt_does_not_leak_evaluation_labels(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 2)
    rendered = render_user_input(package)

    assert "case:2" not in rendered
    assert "accepted_case_ids" not in rendered
    assert "target_present" not in rendered
    assert "evidence_sufficient" not in rendered
    assert "expected_action" not in rendered
    assert prompt_evidence(package)[0]["case_id"] in rendered


def test_sampling_is_deterministic_balanced_and_warm_cold_aware(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    arguments = {
        "modes": ("facts_only", "facts_principle"),
        "dataset": corpus,
        "index": index,
        "case_to_id": case_to_id,
        "per_stratum": 1,
        "retrieval_depth": 5,
        "seed": 42,
    }
    first = select_queries(**arguments)
    second = select_queries(**arguments)

    assert first == second
    assert Counter(item.stratum for item in first) == {
        WARM_SUCCESS: 2,
        WARM_FAILURE: 2,
        COLD: 2,
    }


def test_package_matrix_and_fixed_pilot_cover_all_conditions(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    selected = select_queries(
        modes=("facts_only", "facts_principle"),
        dataset=corpus,
        index=index,
        case_to_id=case_to_id,
        per_stratum=1,
        retrieval_depth=5,
        seed=42,
    )
    packages = build_packages(
        selected,
        top_ks=(1, 3, 5),
        index=index,
        dataset=corpus,
        case_to_id=case_to_id,
    )
    pilot = select_pilot(packages)

    assert len(packages) == 22
    assert len(pilot) == 12
    assert {item.condition for item in pilot} == set(EvidenceCondition)
    assert {item.top_k for item in pilot} == {None, 1, 5}
    assert {item.warm_start for item in pilot} == {True, False}
    assert {item.expected_action for item in pilot} == {
        ExpectedAction.ANSWER,
        ExpectedAction.UNKNOWN_NEEDS_REVIEW,
    }


def test_bm25_scores_are_identical_across_python_hash_seeds() -> None:
    code = """
import json
from sg_legal_rag.retrieval.bm25 import BM25Index
terms = [f"term{i}" for i in range(80)]
documents = [
    " ".join(term for i, term in enumerate(terms) for _ in range((i % 7) + 1)),
    " ".join(terms[::2]),
    " ".join(terms[1::3]),
]
scores = BM25Index(documents).scores(" ".join(terms))
print(json.dumps(scores, sort_keys=True, separators=(",", ":")))
"""
    outputs = []
    for seed in ("1", "2", "77", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                env=environment,
                text=True,
            ).strip()
        )

    assert len(set(outputs)) == 1


def test_repeated_reconstruction_has_identical_evidence_signature(
    corpus: CorpusRepairDataset, case_to_id: dict[str, int]
) -> None:
    def reconstruct() -> dict[str, object]:
        rebuilt_index = BM25Index([item.text for item in corpus.contexts])
        selected = select_queries(
            modes=("facts_only", "facts_principle"),
            dataset=corpus,
            index=rebuilt_index,
            case_to_id=case_to_id,
            per_stratum=1,
            retrieval_depth=5,
            seed=42,
        )
        packages = build_packages(
            selected,
            top_ks=(1, 3, 5),
            index=rebuilt_index,
            dataset=corpus,
            case_to_id=case_to_id,
        )
        return evidence_freeze(packages)

    assert reconstruct() == reconstruct()


def test_evidence_signature_excludes_non_prompt_retrieval_floats(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    item = package.evidence[0]
    assert item.retrieval_score is not None
    perturbed_item = item.model_copy(update={"retrieval_score": item.retrieval_score + 1e-12})
    perturbed_package = package.model_copy(update={"evidence": (perturbed_item,)})

    assert evidence_freeze((package,)) == evidence_freeze((perturbed_package,))


def test_manifest_freezes_evidence_and_rejects_reconstruction_drift(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    selected = select_queries(
        modes=("facts_only", "facts_principle"),
        dataset=corpus,
        index=index,
        case_to_id=case_to_id,
        per_stratum=1,
        retrieval_depth=5,
        seed=42,
    )
    packages = build_packages(
        selected,
        top_ks=(1, 3, 5),
        index=index,
        dataset=corpus,
        case_to_id=case_to_id,
    )
    config = RAGConfig(
        modes=("facts_only", "facts_principle"),
        top_ks=(1, 3, 5),
        queries_per_stratum=1,
        seed=42,
        settings=settings,
        expected_output_tokens=180,
        automatic_retries=0,
        pricing_snapshot_date="2026-08-26",
        manual_review_records=12,
    )
    pilot = select_pilot(packages)
    canary = select_canary(packages)
    methodology_audit = audit_packages(
        packages,
        evidence_cutoff_year=2023,
        test_urls=corpus.test_urls,
    )
    manifest = build_manifest(
        config=config,
        signature="test-signature",
        selected=selected,
        packages=packages,
        pilot=pilot,
        canary=canary,
        methodology_audit=methodology_audit,
    )
    rebuilt_manifest = build_manifest(
        config=config,
        signature="test-signature",
        selected=selected,
        packages=packages,
        pilot=pilot,
        canary=canary,
        methodology_audit=methodology_audit,
    )

    assert manifest["schema_version"] == 3
    assert manifest == rebuilt_manifest
    assert manifest["evidence_freeze"]["signature"]
    assert len(manifest["evidence_freeze"]["packages"]) == len(packages)
    assert all(lock["evidence_digests"] for lock in manifest["evidence_freeze"]["packages"])
    assert_frozen_manifest(manifest, copy.deepcopy(manifest))

    drifted = copy.deepcopy(manifest)
    drifted["evidence_freeze"]["packages"][0]["evidence_digests"][0]["passage_digest"] = "changed"
    with pytest.raises(ValueError, match="frozen evidence signature mismatch"):
        assert_frozen_manifest(manifest, drifted)

    assert methodology_audit["oracle_exact_gold_row_origin_verified"] == 4
    assert methodology_audit["retrieved_historical_corpus_verified"] == 18
    assert methodology_audit["retrieved_gold_or_future_leakage_records"] == 0
    review = sufficiency_review_template(packages)
    assert len(review["records"]) == 18
    assert all(record["review"]["expected_action"] is None for record in review["records"])


def test_output_schema_rejects_invalid_status_contract_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="requires at least one cited claim"):
        GroundedAnswer(
            status="answered",
            recommended_case_id="case:0",
            explanation="unsupported",
            claims=[],
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        GroundedAnswer.model_validate(
            {
                "status": "insufficient_evidence",
                "recommended_case_id": None,
                "explanation": "No evidence.",
                "claims": [],
                "confidence": 0.9,
            }
        )


def test_deterministic_citation_validation_accepts_exact_quote_and_rejects_hallucination(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    valid = validate_answer(package, answer_for(package))
    hallucinated = answer_for(package).model_copy(
        update={
            "explanation": "See [2099] SGCA 999 and case:99.",
            "claims": [
                GroundedClaim(
                    statement="Invented proposition.",
                    evidence_id="E1",
                    supporting_quote="words that are absent",
                )
            ],
        }
    )
    invalid = validate_answer(package, hallucinated)

    assert valid.structurally_valid
    assert valid.citation_correctness == 1
    assert not invalid.structurally_valid
    assert {issue.code for issue in invalid.issues} == {
        "quote_not_verbatim",
        "unseen_case_id",
        "unseen_legal_citation",
    }
    assert invalid.unsupported_claim_rate_proxy == 1


def test_target_presence_and_evidence_sufficiency_are_separate_for_abstention(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = reviewed_package(retrieved_package(corpus, index, case_to_id, 0), sufficient=False)
    abstention = GroundedAnswer(
        status="insufficient_evidence",
        recommended_case_id=None,
        explanation="The supplied evidence does not support a precedent recommendation.",
        claims=[],
    )
    correct = evaluate_record(record_for(package, abstention, settings))
    improper = evaluate_record(record_for(package, answer_for(package), settings))

    assert package.target_present
    assert package.evidence_sufficient is False
    assert package.expected_action is ExpectedAction.ABSTAIN
    assert correct.primary_failure_layer == "5_insufficient_evidence_correct_abstention"
    assert correct.abstention_correct
    assert improper.retrieval_generation_layer is None
    assert improper.abstention_layer == "6_insufficient_evidence_inappropriate_answer"
    assert improper.inappropriate_answer


def test_unreviewed_retrieved_evidence_does_not_score_abstention_from_case_presence(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    abstention = GroundedAnswer(
        status="insufficient_evidence",
        recommended_case_id=None,
        explanation="The supplied evidence does not support a precedent recommendation.",
        claims=[],
    )
    outcome = evaluate_record(record_for(package, abstention, settings))

    assert package.target_present
    assert package.evidence_sufficient is None
    assert package.expected_action is ExpectedAction.UNKNOWN_NEEDS_REVIEW
    assert outcome.primary_failure_layer == "7_evidence_sufficiency_unknown_needs_review"
    assert outcome.abstention_correct is None
    assert outcome.inappropriate_answer is None


def test_provider_failure_is_not_scored_as_an_abstention_failure(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 1)
    failed = record_for(
        package,
        None,
        settings,
        error="BadRequestError: invalid request",
        response_id=None,
    )
    outcome = evaluate_record(failed)
    summary = summarize_records([failed])

    assert failed.result.call_status is ProviderCallStatus.PROVIDER_API_FAILURE
    assert outcome.evaluation_status is EvaluationStatus.PROVIDER_API_FAILURE
    assert outcome.primary_failure_layer == "0_provider_api_failure"
    assert outcome.abstention_layer is None
    assert outcome.abstention_correct is None
    assert outcome.inappropriate_answer is None
    assert summary["provider_api_failures"] == 1
    assert summary["structured_output_failures"] == 0
    assert summary["abstention_recall"] is None
    assert summary["inappropriate_answer_rate"] is None
    assert all_generation_attempts_failed([failed], [failed])


def test_structured_output_failure_has_its_own_status(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    failed = record_for(
        package,
        None,
        settings,
        error="response did not contain a parsed answer",
    )
    outcome = evaluate_record(failed)

    assert failed.result.call_status is ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE
    assert outcome.evaluation_status is EvaluationStatus.STRUCTURED_OUTPUT_FAILURE
    assert outcome.primary_failure_layer == "0_structured_output_failure"
    assert outcome.retrieval_generation_layer is None
    assert outcome.abstention_layer is None


def test_sdk_schema_validation_exception_is_structured_output_failure(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)

    class InvalidStructuredGenerator:
        def generate(self, package, settings):
            GroundedAnswer.model_validate({"status": "answered"})

    record = generate_record(
        InvalidStructuredGenerator(),
        package,
        settings,
        "signature",
    )

    assert record.result.call_status is ProviderCallStatus.STRUCTURED_OUTPUT_FAILURE
    assert evaluate_record(record).evaluation_status is EvaluationStatus.STRUCTURED_OUTPUT_FAILURE


def test_generation_command_failure_requires_at_least_one_usable_result(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    failed = record_for(package, None, settings, error="provider down", response_id=None)
    succeeded = record_for(package, answer_for(package), settings)

    assert all_generation_attempts_failed([failed], [failed, succeeded])
    assert not all_generation_attempts_failed([succeeded], [failed, succeeded])
    assert not all_generation_attempts_failed([], [failed, succeeded])


def test_manually_sufficient_target_present_wrong_precedent_is_generation_layer_two(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = reviewed_package(
        retrieved_package(corpus, index, case_to_id, 0, top_k=2), sufficient=True
    )
    wrong = answer_for(package, recommendation="case:1")
    outcome = evaluate_record(record_for(package, wrong, settings))

    assert package.target_present
    assert not outcome.precedent_correct
    assert outcome.retrieval_generation_layer == "2_target_present_generation_incorrect"


def test_cache_round_trip_and_identity_checks(
    tmp_path,
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    record = record_for(package, answer_for(package), settings)
    path = cache_path(tmp_path, "signature", package.package_id)
    save_record(path, record)

    assert load_record(path, run_signature="signature", package_id=package.package_id) == record
    with pytest.raises(ValueError, match="signature mismatch"):
        load_record(path, run_signature="different", package_id=package.package_id)


def test_mock_responses_client_uses_structured_output_without_network(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0)
    parsed = answer_for(package)
    captured = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    input_tokens_details=SimpleNamespace(cached_tokens=20),
                    output_tokens=30,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=0),
                    total_tokens=130,
                ),
                output_parsed=parsed,
                output_text=parsed.model_dump_json(),
                model="gpt-5.6-luna-2026-08-01",
                id="resp_fake",
            )

    generator = OpenAIResponsesGenerator(SimpleNamespace(responses=FakeResponses()))
    result = generator.generate(package, settings)

    assert captured["text_format"] is GroundedAnswer
    assert captured["text"] == {"verbosity": "low"}
    assert "verbosity" not in captured
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "none"}
    assert result.answer == parsed
    assert result.returned_model == "gpt-5.6-luna-2026-08-01"
    assert result.usage.cached_input_tokens == 20


def test_openai_sdk_serializes_verbosity_inside_text_with_json_schema(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    from openai import BadRequestError, OpenAI

    package = retrieved_package(corpus, index, case_to_id, 0)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": "offline mock stop",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = OpenAI(
        api_key="offline-test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    generator = OpenAIResponsesGenerator(client)

    with pytest.raises(BadRequestError, match="offline mock stop"):
        generator.generate(package, settings)

    assert "verbosity" not in captured
    assert captured["text"]["verbosity"] == "low"
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["reasoning"] == {"effort": "none"}
    assert captured["max_output_tokens"] == 600
    assert captured["store"] is False
