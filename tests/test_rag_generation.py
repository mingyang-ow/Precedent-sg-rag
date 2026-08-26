from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from sg_legal_rag.generation.benchmark import select_pilot
from sg_legal_rag.generation.evaluation import evaluate_record, validate_answer
from sg_legal_rag.generation.evidence import (
    EvidenceCondition,
    EvidencePackage,
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
    ProviderResult,
    TokenUsage,
    cache_path,
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
from sg_legal_rag.retrieval.benchmark import QueryRecord
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
            QueryRecord("objective contract interpretation", {alpha.casefold()}),
            QueryRecord("objective contract unmatched terminology", {beta.casefold()}),
            QueryRecord("objective cold-start duty", {cold.casefold()}),
        ],
        "principle_only": [],
        "facts_principle": [
            QueryRecord("contract facts objective interpretation", {alpha.casefold()}),
            QueryRecord("contract absent principle", {beta.casefold()}),
            QueryRecord("sentencing new cold principle", {cold.casefold()}),
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
        test_urls=frozenset(),
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
        prompt_version="rag-v1",
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
            response_id="resp_test",
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


def test_oracle_context_uses_known_relevant_historical_evidence(
    corpus: CorpusRepairDataset, case_to_id: dict[str, int]
) -> None:
    package = package_oracle_evidence(
        mode="facts_only",
        query=corpus.queries_by_mode["facts_only"][0],
        stratum=WARM_SUCCESS,
        dataset=corpus,
        case_to_id=case_to_id,
    )

    assert package.condition is EvidenceCondition.ORACLE
    assert package.top_k is None
    assert package.evidence[0].case_id == "case:0"
    assert package.retrieval_correct


def test_oracle_context_rejects_cold_query(
    corpus: CorpusRepairDataset, case_to_id: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="warm relevant case"):
        package_oracle_evidence(
            mode="facts_only",
            query=corpus.queries_by_mode["facts_only"][2],
            stratum=COLD,
            dataset=corpus,
            case_to_id=case_to_id,
        )


def test_prompt_does_not_leak_evaluation_labels(
    corpus: CorpusRepairDataset, index: BM25Index, case_to_id: dict[str, int]
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 2)
    rendered = render_user_input(package)

    assert "case:2" not in rendered
    assert "accepted_case_ids" not in rendered
    assert "retrieval_correct" not in rendered
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


def test_correct_abstention_and_inappropriate_answer_have_separate_failure_layers(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 1)
    abstention = GroundedAnswer(
        status="insufficient_evidence",
        recommended_case_id=None,
        explanation="The supplied evidence does not support a precedent recommendation.",
        claims=[],
    )
    correct = evaluate_record(record_for(package, abstention, settings))
    improper = evaluate_record(record_for(package, answer_for(package), settings))

    assert correct.primary_failure_layer == "5_insufficient_evidence_correct_abstention"
    assert correct.abstention_correct
    assert improper.retrieval_generation_layer == (
        "3_retrieval_incorrect_generation_grounded_to_wrong_evidence"
    )
    assert improper.abstention_layer == ("6_insufficient_evidence_inappropriate_answer_or_error")
    assert improper.inappropriate_answer


def test_retrieval_correct_wrong_precedent_is_generation_layer_two(
    corpus: CorpusRepairDataset,
    index: BM25Index,
    case_to_id: dict[str, int],
    settings: GenerationSettings,
) -> None:
    package = retrieved_package(corpus, index, case_to_id, 0, top_k=2)
    wrong = answer_for(package, recommendation="case:1")
    outcome = evaluate_record(record_for(package, wrong, settings))

    assert package.retrieval_correct
    assert not outcome.precedent_correct
    assert outcome.retrieval_generation_layer == "2_retrieval_correct_generation_incorrect"


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
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "none"}
    assert result.answer == parsed
    assert result.returned_model == "gpt-5.6-luna-2026-08-01"
    assert result.usage.cached_input_tokens == 20
