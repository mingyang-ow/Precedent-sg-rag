# Phase 2: BM25 + BGE hybrid baseline

## Objective

Test whether BM25's principle-query strength and BGE-small's fact-query strength are complementary
under one reproducible hybrid candidate generator. This is Experiment C in the project
specification. It uses the same pooled and full-temporal protocols as the component baselines and
does not tune fusion parameters on the test set.

The answer is mostly negative for equal-weight reciprocal rank fusion. The hybrid improves the
weaker component in each full-corpus query mode, but it does not beat the stronger component. It is
therefore a useful rejected baseline, not the retrieval configuration to promote to production.

The run was produced on CPU on 25 August 2026. Raw outputs are committed in
[`experiments/results/`](../experiments/results/).

## Fixed fusion protocol

The dense component is the revision-pinned BGE-small model selected in the Phase 1 comparison.
BM25 retains `k1=1.2` and `b=0.75`. The hybrid uses weighted reciprocal rank fusion (RRF):

```text
score(document) = 1 / (60 + BM25 rank) + 1 / (60 + BGE rank)
```

Both weights are `1.0`, the RRF constant is `60`, and each component contributes at most 1,000
candidates. BGE always contributes its top 1,000. BM25 contributes up to 1,000 positive-score
matches; zero-score documents are excluded so ascending candidate IDs cannot become arbitrary
lexical evidence. The union is ordered by fused score, then by ascending candidate ID.

These choices were fixed before the final hybrid results were inspected. RRF avoids treating BM25
scores and cosine similarity as if they shared a calibrated scale. The complete configuration is
in [`configs/hybrid.toml`](../configs/hybrid.toml).

On the 1,000-way author pools, BGE covers the complete pool, so hybrid MRR is over all candidates.
On the full corpus, MRR is bounded by the two 1,000-deep component lists: a relevant case outside
their union receives zero reciprocal rank. This reflects the configured candidate generator, but
means its MRR has a different retrieval-depth boundary from the earlier exact full-ranking
component diagnostics. Recall@K remains directly comparable.

## Protocol A: authors' 1,000-way pools

The paired comparison uses the same 9,979 principle-pool candidate sets for every query
representation.

| Retriever | Facts MRR | Facts R@10 | Principle MRR | Principle R@10 | Facts + principle MRR | Facts + principle R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1.058% | 1.653% | 2.169% | 2.726% | 2.166% | 2.936% |
| BGE-small | **1.895%** | **3.437%** | **2.406%** | **3.527%** | 2.239% | 3.898% |
| BM25 + BGE RRF | 1.650% | 2.996% | 2.310% | 3.337% | **2.283%** | **3.938%** |

Fusion gives the best combined-query result of these two components, but only by 0.044 percentage
points MRR and 0.040 points Recall@10 over BGE. It dilutes BGE's larger fact-only advantage and
does not surpass BGE on principle-only queries.

## Protocol B: full corpus with temporal test queries

All systems use the same 48,475 candidate strings and grouped 2024–2025 test queries.

| Retriever | Facts MRR | Facts R@10 | Principle MRR | Principle R@10 | Facts + principle MRR | Facts + principle R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.199% | 0.019% | **1.387%** | **1.969%** | **1.149%** | **1.696%** |
| BGE-small | **1.342%** | **0.143%** | 0.671% | 1.106% | 0.401% | 0.672% |
| BM25 + BGE RRF | 0.622% | 0.074% | 0.935% | 1.445% | 0.526% | 0.842% |

Equal RRF consistently lands between BM25 and BGE. It improves fact MRR by 0.423 percentage points
over BM25 and principle MRR by 0.264 points over BGE, showing that both candidate sources add
signal. However, overlap rewarded by equal RRF is not a reliable proxy for relevance in this
candidate representation, and the stronger component is pulled down in every mode.

This result is not a reason to adjust weights against the temporal test. A defensible follow-up is
to select a global weight and candidate depth on the 2022–2023 validation judgments, freeze them,
and evaluate once on 2024–2025. Query-mode-specific weights would be an oracle unless a production
query classifier were independently specified and evaluated.

## CPU latency

Latency excludes model encoding and model/index construction. Dense matrix scoring is amortized
over batches of 64; hybrid latency includes BM25 retrieval, dense top-1,000 selection, candidate
union, and RRF sorting.

| Query mode | Mean | P95 |
|---|---:|---:|
| Facts | 26.36 ms | 37.72 ms |
| Principle | 26.22 ms | 51.70 ms |
| Facts + principle | 25.93 ms | 46.10 ms |

This is slower than either standalone component because fusion ranks a candidate union of up to
2,000 documents per query. It remains an in-process CPU microbenchmark, not end-to-end API
latency.

## Limitations and technical debt

1. Candidate documents remain raw `Cited Case` strings, not precedent facts, holdings, or reasons.
   Fusion cannot recover semantic evidence absent from both component representations.
2. Equal RRF is an intentionally untuned baseline. Weight and depth selection require a
   validation-only experiment before another temporal-test claim.
3. Full-corpus hybrid MRR is candidate-depth bounded, unlike the exact component ranking
   diagnostics. The retrieval cutoff is explicit, but the distinction must be retained in tables.
4. The principle field remains oracle-style extracted input rather than a normal cold-start user
   query.
5. Citation aliases, suspicious target strings, and unavailable canonical candidate dates remain
   unresolved.
6. The best hybrid full-corpus Recall@10 is still below 1.5%. Reranking cannot recover a relevant
   case that is missing from the hybrid candidate union.

The next scoped experiment is reranking the frozen hybrid candidate list. It should report both
candidate recall and reranked quality so a cross-encoder is not credited for upstream misses.

## Reproduction

Install the locked dense dependency set and run both protocols:

```bash
uv sync --locked --extra dev --extra dense
uv run sg-legal-hybrid --protocol pooled \
  --output experiments/results/hybrid_pooled_bm25_bge_rrf.json
uv run sg-legal-hybrid --protocol full \
  --output experiments/results/hybrid_full_bm25_bge_rrf.json
```

Use `--max-queries N` only for smoke tests. The benchmark records a non-null limit in its output so
a capped run cannot be confused with the final metrics.
