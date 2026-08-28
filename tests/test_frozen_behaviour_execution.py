from __future__ import annotations

import json
from pathlib import Path

import pytest

from sg_legal_rag.generation import benchmark as benchmark_module
from sg_legal_rag.generation.adjudication import load_pilot_adjudication
from sg_legal_rag.generation.behaviour_pilot import (
    FrozenBehaviourPackages,
    canonical_digest,
    load_behaviour_adjudication,
    load_behaviour_pilot,
    load_frozen_behaviour_packages,
)
from sg_legal_rag.generation.benchmark import (
    load_config,
    load_json,
    parse_args,
    preflight_frozen_behaviour_execution,
    validate_frozen_behaviour_execution,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/rag_baseline.toml"
MANIFEST_PATH = PROJECT_ROOT / "experiments/samples/rag_baseline.json"
PILOT_PATH = PROJECT_ROOT / "experiments/samples/rag_behaviour_pilot.json"
PACKAGES_PATH = PROJECT_ROOT / "experiments/samples/rag_behaviour_packages.json"
ADJUDICATION_PATH = PROJECT_ROOT / "experiments/samples/rag_behaviour_adjudication.json"
ANSWER_ADJUDICATION_PATH = PROJECT_ROOT / "experiments/samples/rag_pilot_adjudication.json"


def frozen_inputs() -> dict[str, object]:
    return {
        "config": load_config(CONFIG_PATH),
        "global_manifest": load_json(MANIFEST_PATH),
        "pilot": load_behaviour_pilot(PILOT_PATH),
        "adjudication": load_behaviour_adjudication(ADJUDICATION_PATH),
        "answer_adjudication": load_pilot_adjudication(ANSWER_ADJUDICATION_PATH),
        "frozen_packages": load_frozen_behaviour_packages(PACKAGES_PATH),
    }


def test_frozen_preflight_loads_exactly_12_packages_in_frozen_order() -> None:
    _, pilot, packages, labeled_packages = preflight_frozen_behaviour_execution(
        rag_config_path=CONFIG_PATH,
        global_manifest_path=MANIFEST_PATH,
        behaviour_manifest_path=PILOT_PATH,
        behaviour_packages_path=PACKAGES_PATH,
        behaviour_adjudication_path=ADJUDICATION_PATH,
        answer_adjudication_path=ANSWER_ADJUDICATION_PATH,
    )

    package_ids = tuple(package.package_id for package in packages)
    assert len(packages) == len(labeled_packages) == 12
    assert package_ids == pilot.selected_package_ids
    assert package_ids == tuple(package.package_id for package in labeled_packages)


def test_frozen_evidence_digest_verifies_against_global_manifest() -> None:
    inputs = frozen_inputs()

    packages, _ = validate_frozen_behaviour_execution(**inputs)

    assert len(packages) == 12


def test_mutated_evidence_is_rejected_even_with_recomputed_artifact_digest() -> None:
    inputs = frozen_inputs()
    artifact = inputs["frozen_packages"]
    assert isinstance(artifact, FrozenBehaviourPackages)
    first = artifact.packages[0]
    evidence = first.evidence[0].model_copy(update={"passage": first.evidence[0].passage + "x"})
    changed = first.model_copy(update={"evidence": (evidence,) + first.evidence[1:]})
    packages = (changed,) + artifact.packages[1:]
    payload = [package.model_dump(mode="json") for package in packages]
    inputs["frozen_packages"] = artifact.model_copy(
        update={"packages": packages, "package_payload_digest": canonical_digest(payload)}
    )

    with pytest.raises(ValueError, match="evidence or input changed"):
        validate_frozen_behaviour_execution(**inputs)


def test_mutated_package_order_is_rejected() -> None:
    inputs = frozen_inputs()
    artifact = inputs["frozen_packages"]
    assert isinstance(artifact, FrozenBehaviourPackages)
    packages = (artifact.packages[1], artifact.packages[0]) + artifact.packages[2:]
    inputs["frozen_packages"] = artifact.model_copy(update={"packages": packages})

    with pytest.raises(ValueError, match="package IDs or ordering changed"):
        validate_frozen_behaviour_execution(**inputs)


def test_prompt_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = frozen_inputs()
    monkeypatch.setattr(
        benchmark_module,
        "SYSTEM_INSTRUCTIONS",
        benchmark_module.SYSTEM_INSTRUCTIONS + " changed",
    )

    with pytest.raises(ValueError, match="generation contract changed"):
        validate_frozen_behaviour_execution(**inputs)


def test_output_schema_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = frozen_inputs()
    schema = benchmark_module.GroundedAnswer.model_json_schema()
    monkeypatch.setattr(
        benchmark_module.GroundedAnswer,
        "model_json_schema",
        classmethod(lambda cls: {**schema, "changed": True}),
    )

    with pytest.raises(ValueError, match="generation contract changed"):
        validate_frozen_behaviour_execution(**inputs)


def test_wrong_model_or_config_is_rejected() -> None:
    inputs = frozen_inputs()
    config = inputs["config"]
    changed_settings = config.settings.model_copy(update={"model": "wrong-model"})
    inputs["config"] = config.__class__(**{**config.__dict__, "settings": changed_settings})

    with pytest.raises(ValueError, match="generation contract changed"):
        validate_frozen_behaviour_execution(**inputs)


def test_wrong_run_signature_is_rejected() -> None:
    inputs = frozen_inputs()
    pilot = inputs["pilot"]
    inputs["pilot"] = pilot.model_copy(update={"run_signature": "0" * 24})

    with pytest.raises(ValueError, match="run signature mismatch"):
        validate_frozen_behaviour_execution(**inputs)


def test_provider_is_not_constructed_when_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = json.loads(PACKAGES_PATH.read_text(encoding="utf-8"))
    artifact["packages"][0]["evidence"][0]["passage"] += "x"
    artifact["package_payload_digest"] = canonical_digest(artifact["packages"])
    changed_path = tmp_path / "changed-packages.json"
    changed_path.write_text(json.dumps(artifact), encoding="utf-8")
    constructed = False

    def forbidden_provider() -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(benchmark_module, "OpenAIResponsesGenerator", forbidden_provider)

    status = benchmark_module.main(
        [
            "--execute",
            "--behaviour-pilot",
            "--behaviour-packages",
            str(changed_path),
        ]
    )

    assert status == 1
    assert constructed is False


def test_full_reconstruction_verification_has_a_separate_cli_mode() -> None:
    args = parse_args(["--reconstruct-and-verify", "--behaviour-pilot"])

    assert args.reconstruct_and_verify is True
    assert args.execute is False
    assert args.behaviour_pilot is True
