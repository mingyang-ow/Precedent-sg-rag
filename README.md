# Precedent SG RAG

An evaluation-first Singapore legal citation retrieval and grounded RAG project built on
[SG-LegalCite](https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite).

The system is an engineering benchmark, not a legal-advice service. Dataset validation, retrieval
protocols, and failure analysis are treated as first-class outputs.

## Current scope: Phase 3 bounded RAG evaluation

- Reproducible download from an immutable Hugging Face revision.
- Streaming schema, integrity, completeness, and extraction-quality checks.
- Primary chronological split plus a judgment-grouped random comparison split.
- Leakage audit across judgment identifiers, normalized case families, and cited targets.
- Dependency-free BM25 with pooled and full-corpus evaluation protocols.
- Exact cosine retrieval with three revision-pinned English embedding models.
- Deterministic BM25 + BGE weighted reciprocal rank fusion.
- Revision-pinned top-50 TinyBERT cross-encoder reranking ablation.
- Leakage-safe historical citation passages and bounded case profiles.
- Conservative case identity, duplicate removal, and warm/cold coverage audits.
- Restart-safe embedding and query-scoring checkpoints for long experiments.
- Facts-only, principle-only, and facts-plus-principle ablations.
- Recall@K, MRR, nDCG@K, latency, court, and year reporting.
- Deterministic oracle, retrieved-context, and insufficient-evidence generation conditions.
- Strict structured outputs, traceable quotes, explicit abstention, and restart-safe result caches.
- Layered retrieval, generation, citation, hallucination, and abstention evaluation.

## Quick start

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run sg-legal-download
uv run sg-legal-download --include-benchmark
uv run sg-legal-validate --output reports/dataset_validation.json
uv run sg-legal-validate-benchmark --output reports/benchmark_validation.json
uv run sg-legal-split --strategy temporal --output data/processed/splits_temporal.csv
uv run sg-legal-bm25 --protocol pooled --output experiments/results/bm25_pooled.json
uv run sg-legal-bm25 --protocol full --output experiments/results/bm25_full_temporal.json
uv sync --extra dev --extra dense
uv run sg-legal-dense --protocol pooled --model bge_small \
  --output experiments/results/dense_pooled_bge_small.json
uv run sg-legal-dense --protocol full --model bge_small \
  --output experiments/results/dense_full_bge_small.json
uv run sg-legal-hybrid --protocol pooled \
  --output experiments/results/hybrid_pooled_bm25_bge_rrf.json
uv run sg-legal-hybrid --protocol full \
  --output experiments/results/hybrid_full_bm25_bge_rrf.json
uv run sg-legal-rerank --protocol pooled \
  --output experiments/results/reranker_pooled_hybrid_tinybert_l2.json
uv run sg-legal-rerank --protocol full \
  --output experiments/results/reranker_full_hybrid_tinybert_l2.json
uv run sg-legal-corpus-repair --representation passages --retriever bm25 \
  --output experiments/results/corpus_repair_passages_bm25.json
uv sync --extra dev --extra generation
# Offline preparation only; no API request is made without the explicit --execute flag.
uv run sg-legal-rag-evaluate
uv run pytest
```

The default download fetches only the core CSV (about 764 MB). Use
`uv run sg-legal-download --include-benchmark` to also fetch the authors' candidate-pool files.
Raw and processed data are gitignored.

See [data/README.md](data/README.md) for provenance, [docs/dataset.md](docs/dataset.md) for the
validation and split rationale, and [docs/dataset_profile.md](docs/dataset_profile.md) for the
observed Phase 0 findings. The first benchmark and failure analysis are in
[docs/retrieval_baseline.md](docs/retrieval_baseline.md). The dense model comparison and Phase 1
recommendation are in [docs/dense_baseline.md](docs/dense_baseline.md).
The fixed-fusion hybrid experiment and its rejected-baseline analysis are in
[docs/hybrid_baseline.md](docs/hybrid_baseline.md).
The cross-encoder ablation, candidate-recall accounting, and latency tradeoff are in
[docs/reranker_baseline.md](docs/reranker_baseline.md).
The leakage-safe historical-context construction, warm/cold coverage, full repair matrix, and
decision to proceed to bounded RAG evaluation are in
[docs/corpus_repair.md](docs/corpus_repair.md).
The frozen generation protocol, request/cost forecast, output contract, and pre-inference hold are
in [docs/rag_baseline.md](docs/rag_baseline.md).

## Roadmap

1. Dataset validation and leakage analysis. ✓
2. Full-corpus BM25 baseline. ✓
3. Dense retrieval model comparison. ✓
4. Hybrid retrieval and reranking baselines. ✓ Both are rejected for production on the released
   candidate strings; validation-tuned fusion remains optional.
5. Leakage-safe citation-context corpus repair. ✓ Passage BM25 materially improves retrieval;
   cold-start limits remain explicit.
6. Citation-constrained generation and component-level evaluation. ◐ Offline pipeline and frozen
   subset complete; inference awaits explicit cost approval.
7. API, experiment tracking, and observability.

Dataset material is CC BY 4.0 and remains under its upstream licence. Project code is MIT
licensed. Do not treat the project licence as relicensing the dataset or source judgments.
