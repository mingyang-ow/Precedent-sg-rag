# Phase 1: dense retrieval model comparison

## Objective

Compare compact, reproducible dense baselines against BM25 under the same two protocols and query
ablations. This phase asks whether off-the-shelf English embeddings can compensate for the lexical
mismatch between a legal fact or principle and the released candidate representation.

The answer is mixed. Dense retrieval improves fact-only ranking, especially in sampled pools, but
BM25 remains substantially stronger when principles are available in the 48,475-candidate temporal
test. None of the systems has production-ready recall.

These results were produced on CPU from the pinned dataset on 24–25 August 2026. Raw JSON outputs
are committed in [`experiments/results/`](../experiments/results/).

## Models and representation

| Key | Model | Revision | Parameters | Dimensions | Max tokens | Query instruction | Licence |
|---|---|---|---:|---:|---:|---|---|
| `minilm` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4…` | 22.7M | 384 | 256 | None | Apache-2.0 |
| `bge_small` | [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `5c38ec7c405e…` | 33.4M | 384 | 512 | Model-card retrieval prefix | MIT |
| `mpnet` | [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | `e8c3b32edf54…` | 109M | 768 | 384 | None | Apache-2.0 |

Documents are embedded without a prefix. BGE queries use
`Represent this sentence for searching relevant passages: ` as specified by its model card.
Embeddings are L2-normalized and scored by exact dot product, which is cosine similarity. Ties are
broken by ascending candidate ID. No approximate index is used, so quality is independent of ANN
parameters.

Model revisions, dimensions, batch sizes, and the query prefix are pinned in
[`configs/dense_models.toml`](../configs/dense_models.toml). Embedding caches are keyed by model,
revision, role, and a length-prefixed SHA-256 digest of the ordered texts. Cache files and model
weights are gitignored; only metrics are committed.

The released candidates are still raw `Cited Case` strings rather than precedent judgment text.
Dense models therefore encode names, citations, and occasional extraction fragments—not holdings,
facts, or full reasons.

## Protocol A: authors' 1,000-way pools

The paired comparison uses the same 9,979 principle-pool candidate sets for all three query
representations. Each query has one labelled relevant case.

| Model | Facts MRR | Facts R@10 | Principle MRR | Principle R@10 | Facts + principle MRR | Facts + principle R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1.058% | 1.653% | 2.169% | 2.726% | 2.166% | 2.936% |
| MiniLM | 1.932% | 3.638% | **2.545%** | 3.848% | **2.567%** | 4.209% |
| BGE-small | 1.895% | 3.437% | 2.406% | 3.527% | 2.239% | 3.898% |
| MPNet | **2.301%** | **4.069%** | 2.466% | **4.109%** | 2.525% | **4.790%** |

Dense retrieval roughly doubles fact-only Recall@10. MPNet has the strongest pooled recall, while
MiniLM has the best principle and combined MRR. The gain is not large enough to evaluate MPNet by
pooled quality alone: its vectors are twice as wide, and its mean sequential pooled retrieval
latency is about 5–7 ms versus roughly 0.5–0.7 ms for the 384-dimensional models.

## Protocol B: full corpus with temporal test queries

All models rank the same 48,475 unique cited-case strings. Queries come from judgments decided in
2024–2025 and retain the BM25 benchmark's grouped, multi-relevant labels.

| Model | Facts MRR | Facts R@10 | Principle MRR | Principle R@10 | Facts + principle MRR | Facts + principle R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.199% | 0.019% | **1.387%** | **1.969%** | **1.149%** | **1.696%** |
| MiniLM | 0.851% | 0.131% | 0.665% | 1.090% | 0.285% | 0.433% |
| BGE-small | **1.342%** | 0.143% | **0.671%** | **1.106%** | **0.401%** | **0.672%** |
| MPNet | 1.228% | **0.188%** | 0.364% | 0.571% | 0.187% | 0.233% |

BGE-small gives the best dense MRR in all three full-corpus modes and the best fact-only MRR
overall. MPNet retrieves a slightly larger share of the many fact-query relevant cases by rank 10,
but degrades sharply on principles and combined queries. BM25 remains more than twice as strong as
BGE on principle and combined MRR.

The facts-only query count is 740, with 14.84 relevant cases per query on average. Principle-only
has 8,034 queries and facts-plus-principle has 8,057, both averaging about 1.37 relevant cases.
Court and year breakdowns for every model and mode are retained in the result JSON rather than
summarized as causal model differences.

## CPU cost and latency

Initial encoding was measured on an AMD Ryzen 7 7800X3D CPU. For the same 15,546 unique pooled
queries, encoding took about 217 seconds for MiniLM, 468 seconds for BGE-small, and 1,379 seconds
for MPNet. The raw full-corpus float32 vectors require about 71 MiB for either 384-dimensional
model and 142 MiB for MPNet, before ANN or metadata overhead.

Full-corpus retrieval latency excludes model encoding and amortizes batched matrix scoring over 64
queries. It includes exact cosine scoring and deterministic top-k ranking.

| Model | Facts mean / p95 | Principle mean / p95 | Facts + principle mean / p95 |
|---|---:|---:|---:|
| BM25 | 8.67 / 11.71 ms | 13.14 / 30.71 ms | 19.18 / 39.15 ms |
| MiniLM | **1.44 / 3.19 ms** | **1.29 / 3.12 ms** | 1.36 / 3.43 ms |
| BGE-small | 1.86 / 4.38 ms | 1.33 / 3.31 ms | **1.33 / 3.35 ms** |
| MPNet | 1.76 / 5.08 ms | 1.62 / 4.52 ms | 1.55 / 3.57 ms |

Dense scoring is fast here because all embeddings are precomputed and CPU BLAS processes query
batches. These are in-process algorithm microbenchmarks, not end-to-end service latency. Encoding,
model loading, vector-index lookup, serialization, networking, and concurrency are outside this
measurement boundary.

## Recommendation and limitations

Use BGE-small as the dense component for the next hybrid experiment, but retain BM25 rather than
replacing it. BGE has the best dense temporal MRR, 512-token query capacity, 384-dimensional
storage, and much lower encoding cost than MPNet. A score-fusion or candidate-union experiment can
test whether BGE's fact strength complements BM25's principle strength.

Do not promote MPNet as the default. Its modest pooled Recall@10 advantage does not survive as a
broad full-corpus quality win, while it doubles vector storage and has by far the highest encoding
cost. MiniLM remains useful as a fast, compact control.

The Phase 1 systems are not production-ready:

1. Candidates lack precedent facts, holdings, and reasons, which is the dominant information
   bottleneck for both lexical and dense retrieval.
2. These are general English encoders, not Singapore-law models, and no model was fine-tuned on the
   temporal training split.
3. The principle field is extracted with knowledge of the citation context and is an oracle-style
   input, not a cold-start production query.
4. Citation aliases and 584 suspicious target strings remain untreated; semantically identical
   cases can occupy separate candidate IDs.
5. Candidate dates are unavailable as canonical structured fields, so negatives cannot be safely
   filtered to precedents that existed at query time.
6. Even the best full-corpus Recall@10 is below 2% in every query mode. Generation on top of these
   results would frequently lack the labelled authority.

## Reproduction

Install the CPU-only dense dependency set from the lockfile:

```bash
uv sync --locked --extra dev --extra dense
```

Run both protocols for any configured model key:

```bash
uv run sg-legal-dense --protocol pooled --model minilm \
  --output experiments/results/dense_pooled_minilm.json
uv run sg-legal-dense --protocol full --model minilm \
  --output experiments/results/dense_full_minilm.json
```

The other keys are `bge_small` and `mpnet`. Use `--max-queries N` only for smoke tests; never
publish a limited run as a benchmark result.
