# Precedent SG RAG

An evaluation-first Singapore legal citation retrieval and grounded RAG project built on
[SG-LegalCite](https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite).

The system is an engineering benchmark, not a legal-advice service. Phase 0 pins and validates
the dataset before retrieval code is introduced.

## Current scope: Phase 0

- Reproducible download from an immutable Hugging Face revision.
- Streaming schema, integrity, completeness, and extraction-quality checks.
- Primary chronological split plus a judgment-grouped random comparison split.
- Leakage audit across judgment identifiers, normalized case families, and cited targets.

## Quick start

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run sg-legal-download
uv run sg-legal-download --include-benchmark
uv run sg-legal-validate --output reports/dataset_validation.json
uv run sg-legal-validate-benchmark --output reports/benchmark_validation.json
uv run sg-legal-split --strategy temporal --output data/processed/splits_temporal.csv
uv run pytest
```

The default download fetches only the core CSV (about 764 MB). Use
`uv run sg-legal-download --include-benchmark` to also fetch the authors' candidate-pool files.
Raw and processed data are gitignored.

See [data/README.md](data/README.md) for provenance, [docs/dataset.md](docs/dataset.md) for the
validation and split rationale, and [docs/dataset_profile.md](docs/dataset_profile.md) for the
observed Phase 0 findings.

## Roadmap

1. Dataset validation and leakage analysis.
2. Full-corpus BM25 and dense retrieval benchmarks.
3. Hybrid retrieval and reranking ablations.
4. Citation-constrained generation and component-level evaluation.
5. API, experiment tracking, and observability.

Dataset material is CC BY 4.0 and remains under its upstream licence. Project code is MIT
licensed. Do not treat the project licence as relicensing the dataset or source judgments.
