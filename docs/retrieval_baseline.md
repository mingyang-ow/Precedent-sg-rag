# Phase 1: BM25 retrieval baseline

## Objective

Establish a deterministic lexical baseline before introducing embeddings. The benchmark answers
two different questions:

1. How does BM25 compare under the authors' sampled 1,000-way protocol?
2. How much performance remains when every eligible cited-case string is a candidate and test
   queries come from future judgments?

These results were produced from the pinned dataset on 24 August 2026. Raw JSON outputs are
committed in [`experiments/results/`](../experiments/results/).

## Retrieval representation

The released candidate lookup contains raw `Cited Case` strings, not precedent judgment text. Some
are conventional names and citations; others are long extracted passages. This phase deliberately
uses those strings because that is the upstream benchmark representation. It does not use the
query row's citation paragraph as candidate evidence, which would leak the cited case and principle.

BM25 uses:

- NFKC-normalized, case-folded Unicode word tokens;
- Lucene's positive IDF: `log(1 + (N - df + 0.5) / (df + 0.5))`;
- `k1 = 1.2`, `b = 0.75`;
- ascending candidate ID to break equal-score ties;
- no stemming, stop-word removal, or query expansion.

Configuration is pinned in [`configs/bm25.toml`](../configs/bm25.toml). The implementation has no
runtime dependencies outside the Python standard library.

## Protocol A: authors' 1,000-way pools

Each query has one labelled case and 999 random negatives. The authors' fact-only pool is included
for reference. The paired ablation scores three query representations over the same 9,979
principle-pool candidate sets, avoiding the upstream files' non-semantic numeric IDs.

| Query setting | Queries | MRR | R@1 | R@5 | R@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Authors' fact pool | 9,942 | 1.075% | 0.282% | 0.744% | 1.589% | 0.784% |
| Paired facts only | 9,979 | 1.058% | 0.241% | 0.792% | 1.654% | 0.791% |
| Paired principle only | 9,979 | **2.169%** | **1.453%** | **1.934%** | 2.726% | **1.949%** |
| Paired facts + principle | 9,979 | 2.166% | 1.273% | 1.894% | **2.936%** | 1.926% |

The principle approximately doubles MRR. Combining facts and principle gives the best Recall@10,
but principle-only gives slightly better MRR, Recall@1, and nDCG@10. With candidate documents that
are mostly names and citations, facts add many words that have no useful lexical counterpart.

The upstream README reports higher BM25 values (fact MRR 1.8%, principle-augmented MRR 3.2%), but
does not publish a BM25 implementation alongside the retrieval scripts. These results are therefore
labelled by their exact implementation and are not presented as an exact reproduction.

## Protocol B: full corpus with temporal test queries

The candidate corpus contains all 48,475 unique cited-case strings in the common eligible subset.
Test queries come from the 740 judgments decided in 2024–2025. Repeated query representations are
grouped and all their cited cases are treated as relevant, avoiding false negatives when one fact or
principle has multiple valid precedents.

| Query setting | Queries | Relevant/query mean | MRR | R@1 | R@5 | R@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Facts only | 740 | 14.84 | 0.199% | 0.000% | 0.000% | 0.019% | 0.030% |
| Principle only | 8,034 | 1.37 | **1.387%** | **0.758%** | **1.735%** | **1.969%** | **1.438%** |
| Facts + principle | 8,057 | 1.36 | 1.149% | 0.603% | 1.496% | 1.696% | 1.198% |

The facts-only denominator is materially different: one factual summary represents a whole citing
judgment and has 14.84 relevant cases on average (p95 39, maximum 104). MRR considers the first
relevant case; Recall@K divides hits by the entire relevance set.

### Principle-only breakdown by court

| Court | Queries | MRR | R@10 |
|---|---:|---:|---:|
| SGCA | 1,238 | 0.719% | 0.860% |
| SGCAI | 98 | 0.269% | 1.020% |
| SGHC | 5,542 | 1.462% | 2.084% |
| SGHCF | 574 | 3.017% | 4.297% |
| SGHCR | 576 | 0.677% | 1.100% |

Six principle strings span more than one court and are reported as `MIXED` in the JSON rather than
forced into a court category. Court variation is descriptive; it is not evidence that BM25 is
intrinsically better for one court because query and candidate distributions differ.

## Latency

Measured in-process on the development CPU, excluding CSV loading and index construction:

| Protocol and query | p50 | p95 |
|---|---:|---:|
| Pooled facts | 0.80 ms | 0.91 ms |
| Pooled facts + principle | 0.88 ms | 1.06 ms |
| Full facts | 9.47 ms | 11.71 ms |
| Full principle | 12.15 ms | 30.71 ms |
| Full facts + principle | 16.22 ms | 39.15 ms |

These are algorithm microbenchmarks, not API service-level latency. They must not be compared with
future networked or model-serving latency without an end-to-end measurement boundary.

## Failure analysis and implications

1. **Candidate representation is the dominant bottleneck.** Facts and principles are descriptions
   of legal disputes and doctrine, while candidates are usually only names/citations. BM25 cannot
   match semantic relevance that is absent from candidate text.
2. **Facts dilute useful lexical terms.** Principle-only consistently beats the combined query on
   top-rank metrics. This does not show facts are legally unimportant; it shows the corpus lacks
   factual descriptions of candidate precedents.
3. **Sampled pools are optimistic and answer a different question.** A random 1,000-way pool is much
   easier than ranking 48,475 candidates. Results remain in separate tables.
4. **Target identity is noisy.** The Phase 0 audit flagged 584 suspicious targets and 43 duplicate
   semantic labels. Aliases are still treated as separate strings.
5. **The principle is an oracle field.** It was extracted from the known citation context. These
   scores measure principle-aware retrieval, not cold-start deployment from facts alone.
6. **Candidate time cannot be enforced safely.** Cited-case strings do not provide a canonical,
   structured decision date. The shared corpus may include negative cases that post-date a query.
7. **Legal-domain breakdown is unavailable.** The released CSV has `Issue Group` but not the 34
   practice-area tags discussed in the dataset card. Court and test-year breakdowns are reported;
   `Issue Group` is too fine-grained and sparse to relabel as a domain.

## Reproduction

```bash
uv run sg-legal-download --include-benchmark
uv run sg-legal-split --strategy temporal --output data/processed/splits_temporal.csv
uv run sg-legal-bm25 --protocol pooled --output experiments/results/bm25_pooled.json
uv run sg-legal-bm25 --protocol full --output experiments/results/bm25_full_temporal.json
```

Use `--max-queries N` only for smoke tests. Reports produced with that option must not be published
as benchmark results.
