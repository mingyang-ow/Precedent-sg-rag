from __future__ import annotations

import json
from pathlib import Path

from sg_legal_rag.ingestion.benchmark_validation import inspect_benchmark


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_benchmark_audit_checks_pool_integrity_and_pairing(tmp_path: Path) -> None:
    write_json(tmp_path / "stage2_case_lookup.json", {"0": "Case A", "1": "Case B"})
    write_json(
        tmp_path / "stage2_direct_candidate_pools_v2.json",
        {
            "q1": {
                "correct_case_id": 0,
                "correct_case_name": "Case A",
                "fact_text": "facts",
                "pool": [0, 1],
                "pool_size": 2,
            }
        },
    )
    write_json(
        tmp_path / "stage2_single_stage_pools.json",
        {
            "q1": {
                "correct_case_id": 0,
                "correct_case_name": "Case A",
                "fact_text": "facts",
                "principle_text": "principle",
                "query_text": "facts principle",
                "pool": [0, 1],
                "pool_size": 2,
            },
            "q2": {
                "correct_case_id": 1,
                "correct_case_name": "Case B",
                "fact_text": "facts",
                "principle_text": "principle",
                "query_text": "facts principle",
                "pool": [0, 1],
                "pool_size": 2,
            },
        },
    )

    report = inspect_benchmark(tmp_path)

    assert report.lookup_entries == 2
    assert report.direct.pool_sizes == {2: 1}
    assert report.direct.correct_not_in_pool == 0
    assert report.shared_query_ids == 1
    assert report.principle_only_query_ids == 1
