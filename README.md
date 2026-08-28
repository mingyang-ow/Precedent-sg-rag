# Precedent SG RAG

A production-oriented Singapore legal-precedent RAG service with measurable retrieval, explicit
abstention, typed grounded generation, and application-owned citations. Built on
[SG-LegalCite](https://huggingface.co/datasets/anonymousmeowmeow/SG-LegalCite).

The system is a production-oriented RAG portfolio project, not a legal-advice service. Dataset
validation, retrieval protocols, failure analysis, and deterministic evidence traceability are
treated as first-class outputs.

## At a glance

```text
legal facts + optional principle
    -> leakage-safe historical passage BM25
    -> bounded structured generation or explicit abstention
    -> evidence-ID and case-ID validation
    -> application-controlled source passage
    -> typed FastAPI response
```

- Corpus repair improved full-corpus BM25 Recall@10 from about 1.7% to 20.2% by replacing
  identifier-only candidates with historical citation passages.
- A clean-room 12-record behavioral pilot achieved 0.722 balanced accuracy and exposed hidden
  benchmark metadata before expansion.
- Production answers reference immutable evidence; the model never controls displayed source text.
- `/retrieve` works without model credentials; `/answer` uses an injectable provider boundary.
- Startup verifies and restores immutable passage-BM25 artifacts; requests never rebuild the
  source corpus or index.
- Prometheus-compatible metrics track HTTP and RAG phase latency, answer/abstention behavior,
  provider and citation failures, token usage, and estimated cost without user or evidence text.

## Docker quick start

```bash
uv sync --locked --extra generation --extra api
uv run sg-legal-build-retrieval-artifacts
docker build --tag precedent-sg-rag:local .
docker run --rm \
  --mount type=bind,src="$(pwd)/data/processed/retrieval-artifacts",dst=/opt/precedent/retrieval-artifacts,readonly \
  --publish 8000:8000 precedent-sg-rag:local
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
```

Open `http://127.0.0.1:8000/docs`, or call `GET /health`, `GET /ready`, `POST /retrieve`, and
`POST /answer`. Starting the service and checking readiness never call a model provider.

## Current scope: Phase 7 operational observability

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
- Preserved historical quote-based evaluation and restart-safe result caches.
- Versioned production claims that reference immutable evidence IDs instead of generating quotes.
- Application-owned source-text resolution with deterministic ID, visibility, case, and digest
  validation.
- Explicit abstention preserved across the historical and production contracts.
- Typed health, readiness, retrieval, answer, version, and error responses.
- Request IDs, structured privacy-conscious logs, and per-stage latency instrumentation.
- Deterministic, digest-verified corpus and BM25 artifacts loaded without request-time rebuilding.
- Locked, non-root, read-only-capable container with a separately mounted artifact bundle.
- Per-process Prometheus metrics with bounded route, outcome, provider, failure, and citation-code
  labels plus a Grafana-ready dashboard.
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
# Fast integrity check of the frozen 12-record behavioral input artifact; no provider is created.
uv run sg-legal-rag-evaluate --preflight-only --behaviour-pilot
# Slow end-to-end retrieval reconstruction audit; also makes no provider calls.
uv run sg-legal-rag-evaluate --reconstruct-and-verify --behaviour-pilot
# Sanitized blind-review export and cached-output-only clean-room evaluation.
uv run sg-legal-rag-cleanroom --export-review
uv run sg-legal-rag-cleanroom --evaluate
# Compare preserved strict citation metrics with deterministic evaluator-only normalization.
uv run sg-legal-rag-citation-audit
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
The frozen generation protocol, request/cost forecast, historical output contract, and Phase 3
results are in [docs/rag_baseline.md](docs/rag_baseline.md). The application-owned evidence-ID
design is in [docs/production_citation_contract.md](docs/production_citation_contract.md). Service
configuration, endpoints, error mapping, and local startup are in [docs/api.md](docs/api.md).
Artifact construction, failure behavior, Docker usage, and security posture are in
[docs/deployment.md](docs/deployment.md). Metric semantics, privacy constraints, PromQL examples,
and the Grafana dashboard are in [docs/observability.md](docs/observability.md).

## Roadmap

1. Dataset validation and leakage analysis. ✓
2. Full-corpus BM25 baseline. ✓
3. Dense retrieval model comparison. ✓
4. Hybrid retrieval and reranking baselines. ✓ Both are rejected for production on the released
   candidate strings; validation-tuned fusion remains optional.
5. Leakage-safe citation-context corpus repair. ✓ Passage BM25 materially improves retrieval;
   cold-start limits remain explicit.
6. Citation-constrained generation and component-level evaluation. ✓ Frozen behavioral pilot,
   clean-room adjudication, and citation evaluator audit complete.
7. Evidence-ID production citation contract. ✓ Application-owned source resolution and safe
   referential validation complete.
8. FastAPI service. ✓ Typed offline-tested HTTP boundary with injectable provider.
9. Docker plus persistent retrieval artifacts. ✓ Non-root/read-only CI gate complete.
10. Operational observability. ✓ Privacy-safe Prometheus metrics and Grafana-ready dashboard.
11. Security and abuse testing. Next.

Dataset material is CC BY 4.0 and remains under its upstream licence. Project code is MIT
licensed. Do not treat the project licence as relicensing the dataset or source judgments.
