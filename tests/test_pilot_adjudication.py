from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sg_legal_rag.generation.adjudication import (
    PilotAdjudicationRecord,
    adjudication_digest,
    load_pilot_adjudication,
)
from sg_legal_rag.generation.evidence import ExpectedAction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION_PATH = PROJECT_ROOT / "experiments/samples/rag_pilot_adjudication.json"
MANIFEST_PATH = PROJECT_ROOT / "experiments/samples/rag_baseline.json"

EXPECTED_PILOT_PACKAGE_IDS = (
    "0362866e90548d293669f8d3",
    "0ae6a4fcf31bfd744b98cb76",
    "108d21f751d8ec90c0c94ec9",
    "00298e089d66d64c4901df7d",
    "12e71dc2c2cb5a60f8075814",
    "0638f59ac08d1d74fadddce3",
    "054dcfaa3a72933f96aaed04",
    "02bba523326c5266cf09e44e",
    "0928e2d4e249f4350c7cbf49",
    "0f15d4e108bfb9ec51b20fd0",
    "3656dc5505159b1c67a40523",
    "059151ded28fc87c4bbdad98",
)
EXPECTED_ADJUDICATION_DIGEST = "1d603177732e892f150cdffead63e85242b2b865b8e8dcc22dd56182dbc4fd03"
EXPECTED_GLOBAL_EVIDENCE_SIGNATURE = (
    "39d7ce7a0e8a0164712b4dbf1b4fa042b49222c1b6f409f800d0e95805cd29fe"
)
EXPECTED_PILOT_EVIDENCE_SIGNATURE = (
    "8ee6e42f9c4faa12b30a01b70126cb9f406533791daacf2f8f03537295c60eed"
)
EXPECTED_RETRIEVED_PILOT_EVIDENCE_SIGNATURE = (
    "0fb29f47dd2aa58afb60aff85b842d72f92ce6092e9bec304b854e8e9abb3938"
)
EXPECTED_SELECTION_SIGNATURE = "1b0b03f7b6dd90fbead1ce9a622ac88185ce8106354b3a50fcea7dfc09206b6e"
EXPECTED_PACKAGE_ORDER_SIGNATURE = (
    "251afc0253631a505923fa2a52485de7fa8fd36b47714657301744626b4f6917"
)


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_pilot_adjudication_covers_exactly_the_ten_retrieved_packages() -> None:
    adjudication = load_pilot_adjudication(ADJUDICATION_PATH)
    manifest = load_manifest()
    pilot = manifest["pilot"]
    evidence_packages = {
        package["package_id"]: package for package in manifest["evidence_freeze"]["packages"]
    }
    retrieved_ids = tuple(
        package_id
        for package_id in pilot["package_ids"]
        if evidence_packages[package_id]["condition"] == "retrieved_context"
    )

    assert adjudication.pilot_package_ids == EXPECTED_PILOT_PACKAGE_IDS
    assert tuple(record.package_id for record in adjudication.records) == retrieved_ids
    assert len(adjudication.records) == 10


def test_pilot_has_no_unknown_action_after_adjudication() -> None:
    adjudication = load_pilot_adjudication(ADJUDICATION_PATH)
    manifest = load_manifest()
    records_by_id = {record.package_id: record for record in adjudication.records}
    evidence_packages = {
        package["package_id"]: package for package in manifest["evidence_freeze"]["packages"]
    }
    actions = [
        records_by_id[package_id].expected_action
        if package_id in records_by_id
        else ExpectedAction(evidence_packages[package_id]["expected_action"])
        for package_id in manifest["pilot"]["package_ids"]
    ]

    assert len(actions) == 12
    assert ExpectedAction.UNKNOWN_NEEDS_REVIEW not in actions
    assert set(actions) == {ExpectedAction.ANSWER}


def test_target_presence_is_diagnostic_not_the_expected_action_rule() -> None:
    records = load_pilot_adjudication(ADJUDICATION_PATH).records

    assert {record.target_present for record in records} == {True, False}
    assert all(record.expected_action is ExpectedAction.ANSWER for record in records)
    assert all(record.evidence_sufficient for record in records)


def test_adjudication_restricts_actions_to_answer_or_abstain() -> None:
    record = load_pilot_adjudication(ADJUDICATION_PATH).records[0]
    payload = record.model_dump(mode="python")
    payload["expected_action"] = ExpectedAction.UNKNOWN_NEEDS_REVIEW

    with pytest.raises(ValidationError, match="expected_action must reflect evidence_sufficient"):
        PilotAdjudicationRecord.model_validate(payload)


def test_adjudication_digest_is_deterministic_and_frozen() -> None:
    first = load_pilot_adjudication(ADJUDICATION_PATH)
    second = load_pilot_adjudication(ADJUDICATION_PATH)

    assert adjudication_digest(first) == adjudication_digest(second)
    assert adjudication_digest(first) == EXPECTED_ADJUDICATION_DIGEST


def test_manifest_preserves_query_package_and_evidence_integrity() -> None:
    manifest = load_manifest()
    adjudication = load_pilot_adjudication(ADJUDICATION_PATH)
    locks_by_id = {
        package["package_id"]: package for package in manifest["evidence_freeze"]["packages"]
    }
    retrieved_locks = [locks_by_id[record.package_id] for record in adjudication.records]

    assert len(manifest["selection"]["records"]) == 96
    assert canonical_digest(manifest["selection"]) == EXPECTED_SELECTION_SIGNATURE
    assert (
        canonical_digest(manifest["generation_plan"]["package_ids"])
        == EXPECTED_PACKAGE_ORDER_SIGNATURE
    )
    assert tuple(manifest["pilot"]["package_ids"]) == EXPECTED_PILOT_PACKAGE_IDS
    assert manifest["evidence_freeze"]["signature"] == EXPECTED_GLOBAL_EVIDENCE_SIGNATURE
    assert manifest["pilot"]["evidence_signature"] == EXPECTED_PILOT_EVIDENCE_SIGNATURE
    assert canonical_digest(retrieved_locks) == EXPECTED_RETRIEVED_PILOT_EVIDENCE_SIGNATURE
    assert adjudication.pilot_evidence_signature == EXPECTED_PILOT_EVIDENCE_SIGNATURE
    assert (
        adjudication.retrieved_pilot_evidence_signature
        == EXPECTED_RETRIEVED_PILOT_EVIDENCE_SIGNATURE
    )


def test_adjudication_ground_truth_contains_no_model_outputs() -> None:
    payload = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    forbidden = {"answer", "model_output", "raw_output", "response", "response_id"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert not (keys(payload) & forbidden)
    assert payload["blinded_to_model_outputs"] is True
