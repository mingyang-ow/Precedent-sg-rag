# Phase 2: hybrid + cross-encoder reranking

## Objective

Test Experiment D: whether a cross-encoder can improve the frozen BM25 + BGE hybrid by reranking
its top 50 candidates. The experiment reports candidate recall separately from reranked quality so
the reranker is not credited for upstream misses.

The answer is no for the released candidate representation. Reranking reduces MRR in every pooled
and full-temporal query mode, and reduces Recall@10 everywhere except for a negligible fact-only
full-corpus increase. This is a rejected baseline, not a component to promote into a RAG system.

The run was produced on CPU on 25 August 2026. Raw outputs are committed in
[`experiments/results/`](../experiments/results/).

## Model and fixed protocol

The reranker is
[`cross-encoder/ms-marco-TinyBERT-L2-v2`](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)
at revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`. It is an Apache-2.0,
4.39-million-parameter English cross-encoder trained for MS MARCO passage ranking. The model takes
the raw query and raw `Cited Case` string as a pair, truncates the combined input to 512 tokens,
and returns one raw logit. Higher logits rank first, with ascending candidate ID as the tie-break.

The full protocol is frozen in [`configs/reranker.toml`](../configs/reranker.toml):

- equal-weight BM25 + BGE RRF supplies the top 50 from its 1,000-deep component lists;
- TinyBERT reranks all 50 candidates with CPU batch size 32;
- pre/post metrics use the exact same 50-candidate set;
- a relevant case absent from the top 50 receives zero reciprocal rank;
- score caches include the model revision, maximum length, corpus digest, ordered queries, and
  candidate-ID matrix.

TinyBERT-L2 was selected for CPU practicality. The official Sentence Transformers
[MS MARCO comparison](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html) reports lower
MS MARCO MRR@10 than MiniLM-L6 (32.56 versus 39.01), but five times its V100 throughput. On this
CPU, a 512-pair representative check measured about 750 pairs/second for TinyBERT-L2 versus 43 for
MiniLM-L6. The smaller model keeps the uncapped experiment tractable, with an explicit quality
tradeoff.

## Protocol A: authors' 1,000-way pools

The paired table uses the same 9,979 pools in every query mode. The hybrid row is recomputed with
MRR bounded to its top 50 so the before/after comparison is fair; it is therefore lower than the
full-union hybrid MRR in the Experiment C report.

| Query | Hybrid MRR | Reranked MRR | Hybrid R@10 | Reranked R@10 | Candidate R@50 |
|---|---:|---:|---:|---:|---:|
| Facts | **1.331%** | 1.018% | **2.996%** | 2.465% | 10.402% |
| Principle | **1.993%** | 1.491% | **3.337%** | 2.966% | 9.320% |
| Facts + principle | **1.959%** | 1.096% | **3.938%** | 3.157% | 11.825% |

Reranking hurts all paired comparisons. Because each pooled query has one relevant case,
candidate Recall@50 is also the fraction of queries whose labelled case is available to the
cross-encoder. Roughly 88–91% of queries are unrecoverable at this reranking depth.

## Protocol B: full corpus with temporal test queries

The full protocol uses 48,475 candidate strings and grouped 2024–2025 test queries.

| Query | Hybrid MRR | Reranked MRR | Hybrid R@10 | Reranked R@10 | Candidate R@50 | Query hit rate@50 |
|---|---:|---:|---:|---:|---:|---:|
| Facts | **0.475%** | 0.241% | 0.074% | **0.078%** | 0.377% | 4.459% |
| Principle | **0.914%** | 0.621% | **1.445%** | 0.848% | 2.423% | 2.763% |
| Facts + principle | **0.498%** | 0.148% | **0.842%** | 0.185% | 2.214% | 2.532% |

The fact Recall@10 increase is 0.004 percentage points and comes with a 0.234-point MRR loss. The
principle and combined degradations are much larger. TinyBERT was trained to judge query-passage
relevance, but the released candidate is usually only a case name, citation, or extraction
fragment. Its score cannot reliably recover facts, holdings, or legal reasoning that are absent
from the input.

Candidate recall is macro-averaged over queries. The fact hit rate is higher than fact Recall@50
because fact queries average 14.84 relevant cases: the candidate list may contain one relevant
case while missing most of the labelled set. Principle-bearing queries average about 1.37 relevant
cases, yet only 2.5–2.8% have any relevant top-50 candidate.

## CPU cost and latency

Cross-encoder inference is measured once per complete mode and amortized across its queries. The
reported total mean adds this amortized inference cost to measured hybrid candidate generation and
final sorting. The P95 is explicitly named `total_with_amortized_inference` in JSON: it is not a
direct per-query cross-encoder latency sample.

| Protocol / query | Pairs/second | Amortized total mean | Amortized total P95 |
|---|---:|---:|---:|
| Pooled facts | 1,222.62 | 43.79 ms | 44.89 ms |
| Pooled principle | 687.26 | 76.61 ms | 78.76 ms |
| Pooled facts + principle | 430.76 | 120.11 ms | 122.17 ms |
| Full facts | 320.48 | 182.53 ms | 193.15 ms |
| Full principle | 275.50 | 215.09 ms | 237.71 ms |
| Full facts + principle | 636.44 | 94.48 ms | 108.42 ms |

The full uncapped inference phases took about 1 minute 55 seconds for facts, 24 minutes 16 seconds
for principles, and 10 minutes 31 seconds for combined queries. Pooled paired inference took about
6 minutes 44 seconds, 12 minutes 1 second, and 19 minutes 14 seconds respectively. Dynamic padding
and the deterministic query order produce substantial throughput variation by sequence length.

## Decision, limitations, and technical debt

Do not add this reranker to the serving design. It increases CPU cost while degrading the primary
quality metrics.

1. The dominant blocker is document representation. Case identifiers are not passages and do not
   contain the facts, holdings, or reasons the reranker needs.
2. Top-50 candidate hit rates are below 5% in every full-temporal mode. No reranker can recover a
   case missing from its candidate list.
3. TinyBERT-L2 trades effectiveness for CPU throughput and is not trained on Singapore law.
   Testing a larger general reranker before fixing the corpus would spend much more compute on the
   same missing-information problem.
4. The principle field is still oracle-style extracted input rather than a normal production
   query.
5. Per-query cross-encoder P95 requires instrumenting individual inference batches or serving
   requests; this benchmark only reports an explicitly amortized estimate.
6. Alias resolution, suspicious target strings, and candidate-date filtering remain unresolved.

Proceeding directly to generated legal answers would be misleading: even before reranking, full
Recall@10 is below 1.5%, and the available candidate strings are not evidence passages. The next
architectural decision should be whether to ingest licensed full judgment text and map citations
to passage-bearing cases, or to keep this repository strictly as an honest citation-identifier
benchmark.

## Reproduction

Install the locked dense dependencies and run both protocols:

```bash
uv sync --locked --extra dev --extra dense
uv run sg-legal-rerank --protocol pooled \
  --output experiments/results/reranker_pooled_hybrid_tinybert_l2.json
uv run sg-legal-rerank --protocol full \
  --output experiments/results/reranker_full_hybrid_tinybert_l2.json
```

The first run writes gitignored score caches under `data/processed/reranker_scores/`. Later runs
validate and reuse the exact score matrices while preserving the original inference timing.
`--max-queries N` is for smoke tests only and is recorded in the output.
