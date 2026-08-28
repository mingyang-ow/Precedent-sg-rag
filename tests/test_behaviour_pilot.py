from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sg_legal_rag.generation.adjudication import (
    adjudication_digest,
    load_pilot_adjudication,
)
from sg_legal_rag.generation.behaviour_pilot import (
    behaviour_adjudication_digest,
    behaviour_run_signature,
    canonical_digest,
    deterministic_order,
    load_behaviour_adjudication,
    load_behaviour_pilot,
)
from sg_legal_rag.generation.evidence import EvidenceCondition, ExpectedAction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION_PATH = PROJECT_ROOT / "experiments/samples/rag_behaviour_adjudication.json"
PILOT_PATH = PROJECT_ROOT / "experiments/samples/rag_behaviour_pilot.json"
ANSWER_ADJUDICATION_PATH = PROJECT_ROOT / "experiments/samples/rag_pilot_adjudication.json"
MANIFEST_PATH = PROJECT_ROOT / "experiments/samples/rag_baseline.json"

EXPECTED_ANSWER_ADJUDICATION_DIGEST = (
    "1d603177732e892f150cdffead63e85242b2b865b8e8dcc22dd56182dbc4fd03"
)
EXPECTED_ANSWER_PILOT_EVIDENCE_DIGEST = (
    "8ee6e42f9c4faa12b30a01b70126cb9f406533791daacf2f8f03537295c60eed"
)
EXPECTED_GLOBAL_EVIDENCE_DIGEST = "39d7ce7a0e8a0164712b4dbf1b4fa042b49222c1b6f409f800d0e95805cd29fe"
EXPECTED_CANDIDATE_POOL_DIGEST = "a5217bfcfa43093d8ec2cbef9ad22ab0c52042ec4f20d761b2405962cb19611c"
EXPECTED_CANDIDATE_ORDER_DIGEST = "cdc1ad8fb986dd8c52ffbd3221283986be8ad57ad74e56515635e81412b2d99f"
EXPECTED_ADJUDICATION_DIGEST = "5bfca978eef01713f937a08f9212a79aa01a928fb2b0d17ee89f687ce0ba9a15"
EXPECTED_EVIDENCE_DIGEST = "9faf464cb462aa3a4b87a13942f7bea4f7c81cba6db99a97ea8a165aca5cebb5"
EXPECTED_RUN_SIGNATURE = "3664b44b7d4dbe620225d598"


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_behaviour_pilot_has_exact_balanced_composition() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    pilot = load_behaviour_pilot(PILOT_PATH)
    records_by_id = {
        record.package_id: record
        for record in adjudication.reviewed_candidates + adjudication.oracle_reviews
    }
    selected = [records_by_id[package_id] for package_id in pilot.selected_package_ids]

    assert len(selected) == 12
    assert Counter(record.expected_action for record in selected) == {
        ExpectedAction.ANSWER: 6,
        ExpectedAction.ABSTAIN: 6,
    }
    assert Counter(record.query_mode for record in selected) == {
        "facts_only": 6,
        "facts_principle": 6,
    }
    assert Counter(record.condition for record in selected) == {
        EvidenceCondition.ORACLE_GOLD: 2,
        EvidenceCondition.RETRIEVED: 10,
    }
    assert all(record.evidence_sufficient is not None for record in selected)
    assert all(
        record.expected_action is not ExpectedAction.UNKNOWN_NEEDS_REVIEW for record in selected
    )
    assert Counter(
        "oracle" if record.top_k is None else str(record.top_k) for record in selected
    ) == {"oracle": 2, "1": 4, "3": 5, "5": 1}
    assert pilot.counts == {
        "conditions": {"oracle_gold_context": 2, "retrieved_context": 10},
        "expected_actions": {"abstain": 6, "answer": 6},
        "query_modes": {"facts_only": 6, "facts_principle": 6},
        "top_k": {"1": 4, "3": 5, "5": 1, "oracle": 2},
    }


def test_reviewed_retrieved_candidates_are_exact_blind_order_prefix() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    manifest = load_manifest()
    answer_adjudication = load_pilot_adjudication(ANSWER_ADJUDICATION_PATH)
    old_reviewed = {record.package_id for record in answer_adjudication.records}
    candidate_pool = tuple(
        lock["package_id"]
        for lock in manifest["evidence_freeze"]["packages"]
        if lock["condition"] == "retrieved_context"
        and lock["expected_action"] == "unknown_needs_review"
        and lock["package_id"] not in old_reviewed
    )
    reproduced_order = deterministic_order(
        candidate_pool,
        seed=adjudication.seed,
        tag="retrieved",
    )
    reviewed = tuple(
        sorted(adjudication.reviewed_candidates, key=lambda record: record.review_order)
    )

    assert len(candidate_pool) == 278
    assert len(reviewed) == 52
    assert tuple(record.review_order for record in reviewed) == tuple(range(1, 53))
    assert tuple(record.package_id for record in reviewed) == reproduced_order[:52]
    assert adjudication.candidate_order == reproduced_order
    assert canonical_digest(candidate_pool) == EXPECTED_CANDIDATE_POOL_DIGEST
    assert canonical_digest(reproduced_order) == EXPECTED_CANDIDATE_ORDER_DIGEST


def test_retrieved_selection_uses_first_observed_records_for_each_quota() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    reviewed = sorted(adjudication.reviewed_candidates, key=lambda record: record.review_order)
    quotas = {
        ("facts_only", ExpectedAction.ANSWER): 2,
        ("facts_only", ExpectedAction.ABSTAIN): 3,
        ("facts_principle", ExpectedAction.ANSWER): 2,
        ("facts_principle", ExpectedAction.ABSTAIN): 3,
    }
    first_observed: list[str] = []
    for key, count in quotas.items():
        matching = [
            record.package_id
            for record in reviewed
            if (record.query_mode, record.expected_action) == key
        ]
        first_observed.extend(matching[:count])

    selected_retrieved = {record.package_id for record in reviewed if record.selected_for_pilot}
    assert selected_retrieved == set(first_observed)
    assert reviewed[-1].package_id == "404920c168c10748e0258895"
    assert reviewed[-1].expected_action is ExpectedAction.ABSTAIN
    assert reviewed[-1].selected_for_pilot


def test_selected_package_order_reproduces_from_frozen_policy() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    pilot = load_behaviour_pilot(PILOT_PATH)
    retrieved = sorted(adjudication.reviewed_candidates, key=lambda record: record.review_order)
    oracle_by_mode = {
        record.query_mode: record.package_id
        for record in adjudication.oracle_reviews
        if record.selected_for_pilot
    }
    expected: list[str] = []
    for mode in ("facts_only", "facts_principle"):
        expected.append(oracle_by_mode[mode])
        expected.extend(
            record.package_id
            for record in retrieved
            if record.selected_for_pilot
            and record.query_mode == mode
            and record.expected_action is ExpectedAction.ANSWER
        )
        expected.extend(
            record.package_id
            for record in retrieved
            if record.selected_for_pilot
            and record.query_mode == mode
            and record.expected_action is ExpectedAction.ABSTAIN
        )

    assert tuple(expected) == adjudication.selected_package_ids
    assert tuple(expected) == pilot.selected_package_ids


def test_oracle_evidence_is_manually_reviewed_not_mechanically_assumed() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    manifest = load_manifest()
    answer_adjudication = load_pilot_adjudication(ANSWER_ADJUDICATION_PATH)
    oracle_by_id = {record.package_id: record for record in adjudication.oracle_reviews}

    unrelated = oracle_by_id["19b054ad8c323f6da689f55c"]
    fallback = oracle_by_id["8c13b526e62500c845aa95b9"]
    assisted = oracle_by_id["7cfe4dcc0cc44a18b4f6a498"]
    assert unrelated.target_present and not unrelated.evidence_sufficient
    assert unrelated.expected_action is ExpectedAction.ABSTAIN
    assert not unrelated.selected_for_pilot
    assert fallback.evidence_sufficient and fallback.selected_for_pilot
    assert assisted.evidence_sufficient and assisted.selected_for_pilot

    modes_by_query = {
        record["query_id"]: record["mode"] for record in manifest["selection"]["records"]
    }
    fallback_pool = tuple(
        lock["package_id"]
        for lock in manifest["evidence_freeze"]["packages"]
        if lock["condition"] == "oracle_gold_context"
        and modes_by_query[lock["query_id"]] == "facts_only"
        and lock["package_id"] not in answer_adjudication.pilot_package_ids
        and lock["package_id"] not in adjudication.oracle_initial_package_ids
    )
    frozen_fallback_order = deterministic_order(
        fallback_pool,
        seed=adjudication.seed,
        tag="oracle-fallback",
        mode="facts_only",
        separator="\\0",
    )
    assert frozen_fallback_order[0] == fallback.package_id


def test_behaviour_digests_and_run_signature_are_deterministic_and_frozen() -> None:
    adjudication = load_behaviour_adjudication(ADJUDICATION_PATH)
    pilot = load_behaviour_pilot(PILOT_PATH)
    manifest = load_manifest()
    locks_by_id = {lock["package_id"]: lock for lock in manifest["evidence_freeze"]["packages"]}
    selected_locks = [locks_by_id[package_id] for package_id in pilot.selected_package_ids]
    reviews_by_id = {
        record.package_id: record
        for record in adjudication.reviewed_candidates + adjudication.oracle_reviews
    }

    assert behaviour_adjudication_digest(adjudication) == EXPECTED_ADJUDICATION_DIGEST
    assert pilot.adjudication_digest == EXPECTED_ADJUDICATION_DIGEST
    assert canonical_digest(selected_locks) == EXPECTED_EVIDENCE_DIGEST
    assert pilot.evidence_digest == EXPECTED_EVIDENCE_DIGEST
    assert all(
        reviews_by_id[package_id].top_k == locks_by_id[package_id]["top_k"]
        for package_id in pilot.selected_package_ids
    )
    assert (
        behaviour_run_signature(
            global_run_signature=pilot.global_run_signature,
            adjudication_digest_value=pilot.adjudication_digest,
            evidence_digest=pilot.evidence_digest,
        )
        == EXPECTED_RUN_SIGNATURE
        == pilot.run_signature
    )


def test_answer_only_pilot_and_global_evidence_remain_unchanged() -> None:
    answer_adjudication = load_pilot_adjudication(ANSWER_ADJUDICATION_PATH)
    pilot = load_behaviour_pilot(PILOT_PATH)
    manifest = load_manifest()

    assert adjudication_digest(answer_adjudication) == EXPECTED_ANSWER_ADJUDICATION_DIGEST
    assert answer_adjudication.pilot_evidence_signature == EXPECTED_ANSWER_PILOT_EVIDENCE_DIGEST
    assert manifest["evidence_freeze"]["signature"] == EXPECTED_GLOBAL_EVIDENCE_DIGEST
    assert pilot.answer_pilot_adjudication_digest == EXPECTED_ANSWER_ADJUDICATION_DIGEST
    assert pilot.answer_pilot_evidence_digest == EXPECTED_ANSWER_PILOT_EVIDENCE_DIGEST
    assert pilot.global_evidence_digest == EXPECTED_GLOBAL_EVIDENCE_DIGEST


def test_adjudication_contains_no_model_outputs() -> None:
    payload = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    forbidden = {"answer", "model_output", "raw_output", "response", "response_id"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert payload["blinded_to_model_outputs"] is True
    assert not (keys(payload) & forbidden)
