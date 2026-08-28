from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from sg_legal_rag.retrieval.artifacts import RetrievalBuildProvenance, write_retrieval_bundle
from sg_legal_rag.retrieval.bm25 import BM25Index
from sg_legal_rag.retrieval.corpus_repair import CorpusRepairDataset, HistoricalContext


def _context(case_name: str, text: str, position: int) -> HistoricalContext:
    return HistoricalContext(
        case_key=case_name.casefold(),
        raw_case=case_name,
        source_url=f"https://example.test/judgment-{position}",
        source_reference=f"[2023] SGHC {position}",
        source_year=2023,
        text=text,
        original_chars=len(text),
        identifier_matched=True,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = ("Alpha v Beta [2020] SGCA 2", "Crown v Delta [2019] SGHC 4")
    contexts = (
        _context(cases[0], "Alpha v Beta applied objective contract interpretation.", 1),
        _context(cases[1], "Crown v Delta addressed proportional criminal sentencing.", 2),
    )
    dataset = CorpusRepairDataset(
        case_keys=tuple(value.casefold() for value in cases),
        case_texts=cases,
        contexts=contexts,
        context_case_ids=np.asarray([0, 1], dtype=np.int64),
        profiles=cases,
        historical_case_ids=frozenset({0, 1}),
        queries_by_mode={},
        audit={},
        test_urls=frozenset(),
        max_passage_chars=4000,
    )
    write_retrieval_bundle(
        dataset=dataset,
        index=BM25Index([context.text for context in contexts]),
        provenance=RetrievalBuildProvenance(
            dataset_id="ci/synthetic",
            dataset_revision="fixture-v1",
            source_file="fixture.csv",
            source_digest="1" * 64,
            source_size=1,
            config_digest="2" * 64,
            splits_digest="3" * 64,
        ),
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
