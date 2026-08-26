# Phase 2.5: leakage-safe citation-context corpus repair

## Decision: CONTINUE TO RAG

Historical citation passages materially change candidate retrieval. On the 2024–2025 temporal test,
BM25 over leakage-safe passages raises combined-query MRR from 1.196% to 12.517%, Recall@10 from
1.696% to 20.195%, and Recall@50 from 1.988% to 30.932%. On warm-start queries, passage BM25 reaches
18.837% MRR, 32.328% Recall@10, and 49.642% Recall@50.

This is enough to proceed to a bounded grounded-generation and RAG-evaluation phase. It is not
evidence that retrieval is comprehensive or suitable for unsupervised legal advice. Historical
citation contexts describe how an earlier judgment was discussed by later courts; they are not the
full reasons or an authoritative substitute for the cited judgment. Cold-start precedents still
lack semantic candidate evidence.

## Why the corpus needed repair

The original full-corpus benchmark ranked 48,475 raw `Cited Case` values. Most candidates contained
only a case name or citation, while queries contained facts and legal principles. Better embeddings,
fusion, and reranking could not recover information absent from the candidate text.

This phase uses only the pinned SG-LegalCite release and its documented CC BY 4.0 material. No full
judgments or external corpora were downloaded or scraped.

One schema assumption proved incorrect during implementation: `Paragraph` is not reliably a short
citation paragraph. Across the release its median length is 6,081 characters, p95 is 10,976, and
the maximum is 33,426. The implementation therefore extracts a bounded 2,000-character window
centred on the cited identifier when it can locate that identifier, falling back to a deterministic
prefix otherwise. The identifier was found in 95.169% of eligible historical contexts.

## Leakage-safe construction

The main protocol uses one fixed historical evidence cutoff:

```text
judgments dated 2000–2023 -> candidate evidence
judgments dated 2024–2025 -> test queries and labels only
```

SG-LegalCite provides the citing judgment's year, but not a reliable date for every candidate
precedent or a full within-year chronology. A per-query cutoff cannot therefore be implemented
reliably. Allowing 2024 contexts for 2025 queries would also let held-out judgments contribute
candidate text and create cross-query leakage. The conservative fixed cutoff avoids both problems.

The candidate representations are:

| Representation | Source fields | Deterministic policy |
|---|---|---|
| Identifier control | `Cited Case` | One conservative canonical identifier per case |
| Historical passages | `Paragraph`, `Cited Case`, `Year`, `Judgment_URL`, `Judgment_Reference` | One bounded window per unique historical context; case score is the maximum passage score |
| Historical profile | Same fields as passages | Identifier plus at most three contexts, ordered by year descending then URL, reference, and digest; 550 characters per context and 2,000 total |

Duplicate contexts are removed per case using SHA-256 of NFKC-normalized, whitespace-collapsed
text. Profiles cap identifiers at 300 characters; 296 identifiers required that cap. A case with no
historical context receives identifier-only text in the profile representation and is absent from
the passage representation.

Canonical identity is deliberately conservative: NFKC normalization, whitespace collapse, and
case-folding only. This reduces 48,475 eligible raw identifiers to 47,813 canonical identities,
collapsing 662 formatting/case variants. A heuristic audit records 1,440 possible case-family alias
groups but does not fuzzy-merge them. This avoids silently combining distinct proceedings.

## Corpus and coverage audit

| Measure | Value |
|---|---:|
| SG-LegalCite rows | 100,890 |
| Historical rows at or before 2023 | 89,893 |
| Unique historical contexts after deduplication | 89,795 |
| Duplicate historical contexts removed | 98 |
| Canonical cases with historical context | 44,426 |
| Canonical candidate identities | 47,813 |
| Unique 2024–2025 test targets | 6,778 |
| Warm-start unique targets | 3,391 (50.030%) |
| Cold-start unique targets | 3,387 (49.970%) |

A target is warm-start when at least one usable citation context exists at or before the cutoff. A
query is warm-start when at least one of its labelled targets is warm. Warm-query metrics retain
only retrievable warm labels in the relevance denominator; all-query metrics retain every label,
including legitimate zero credit for cold targets absent from the passage corpus.

| Query mode | All queries | Warm queries | Warm-query coverage |
|---|---:|---:|---:|
| Facts only | 740 | 705 | 95.270% |
| Principle only | 8,034 | 5,338 | 66.443% |
| Facts + principle | 8,057 | 5,354 | 66.452% |

Facts are grouped at citing-judgment level and average 14.84 labelled precedents per query, while
principle-bearing modes average about 1.37. Their Recall@K values therefore have materially
different denominators.

## Results

All runs are uncapped exact full-corpus evaluations. Percentages below are macro-averaged over
queries. Full MRR, Recall@1/5/10/20/50, nDCG at each cutoff, latency, court, and year breakdowns are
retained in the linked JSON outputs.

### Facts + principle: all test queries

| Representation | Retriever | MRR | R@10 | R@50 |
|---|---|---:|---:|---:|
| Identifier | BM25 | 1.196% | 1.696% | 1.988% |
| Identifier | BGE-small | 0.403% | 0.672% | 1.458% |
| Historical passages | BM25 | **12.517%** | **20.195%** | **30.932%** |
| Historical passages | BGE-small | 7.541% | 13.161% | 25.596% |
| Historical profile | BM25 | 8.530% | 12.648% | 20.857% |
| Historical profile | BGE-small | 4.229% | 6.553% | 13.863% |

### Facts + principle: warm-start queries

| Representation | Retriever | MRR | R@10 | R@50 |
|---|---|---:|---:|---:|
| Identifier | BM25 | 1.396% | 2.068% | 2.435% |
| Identifier | BGE-small | 0.418% | 0.677% | 1.481% |
| Historical passages | BM25 | **18.837%** | **32.328%** | **49.642%** |
| Historical passages | BGE-small | 11.349% | 21.072% | 40.925% |
| Historical profile | BM25 | 12.713% | 20.034% | 33.100% |
| Historical profile | BGE-small | 6.302% | 10.414% | 22.049% |

### Winning representation by query mode

| Query mode | Cohort | MRR | R@10 | R@50 |
|---|---|---:|---:|---:|
| Facts only | All | 16.092% | 5.029% | 12.660% |
| Facts only | Warm | 16.891% | 7.921% | 20.128% |
| Principle only | All | 13.007% | 20.327% | 30.884% |
| Principle only | Warm | 19.576% | 32.643% | 49.690% |
| Facts + principle | All | 12.517% | 20.195% | 30.932% |
| Facts + principle | Warm | 18.837% | 32.328% | 49.642% |

These rows all use passage BM25. The canonical identifier controls closely reproduce the earlier
raw-identifier baselines, so conservative identity normalization is not the cause of the gain.
Individual passages outperform profiles for both BM25 and BGE: selecting the best local context
preserves useful evidence that a three-context, 2,000-character profile compresses away. BM25 also
outperforms BGE on the main principle and combined tasks, so the selected production candidate
generator is passage BM25 with maximum passage-to-case aggregation.

The optional structured-metadata ablation was not run. Principle, issue, and issue-group fields are
extracted signals with substantial oracle risk, and passage text already answered the phase's
representation question. The previously rejected fixed hybrid was not repeated, and reranking was
not added to this repair phase. Passage BM25 now makes a future passage-aware reranker meaningful
on warm queries, but it is a downstream optimization rather than evidence needed for this decision
gate.

## RAG boundary and remaining limitations

The next phase should use retrieved passage text and its source judgment metadata, evaluate
claim-level faithfulness and citation correctness, and report warm/cold coverage. It should abstain
or disclose insufficient corpus coverage when a target has no historical evidence. Facts-only
evaluation must remain prominent because extracted principles are not always available in a real
user request.

Known limitations remain:

- nearly half of unique test targets are cold-start;
- conservative canonicalization leaves likely aliases unresolved;
- the fixed cutoff cannot model exact within-year availability;
- max-score aggregation may favor a single unusually lexical context;
- citation contexts are secondary descriptions and may omit or simplify the cited judgment;
- profile length and context-count limits are engineering baselines, not legally validated choices;
- latency is an in-process CPU measurement and excludes ingestion, model loading, and serving.

## Reproduction

After downloading the pinned dataset and creating the temporal split, run:

```bash
uv run sg-legal-corpus-repair --representation identifier --retriever bm25 \
  --output experiments/results/corpus_repair_identifier_bm25.json
uv run sg-legal-corpus-repair --representation passages --retriever bm25 \
  --output experiments/results/corpus_repair_passages_bm25.json
uv run sg-legal-corpus-repair --representation profile --retriever bm25 \
  --output experiments/results/corpus_repair_profile_bm25.json
uv run sg-legal-corpus-repair --representation identifier --retriever bge \
  --output experiments/results/corpus_repair_identifier_bge.json
uv run sg-legal-corpus-repair --representation passages --retriever bge \
  --output experiments/results/corpus_repair_passages_bge.json
uv run sg-legal-corpus-repair --representation profile --retriever bge \
  --output experiments/results/corpus_repair_profile_bge.json
```

Expensive embedding and query scoring work is cached. First-time embedding writes an atomic partial
cache after every four model batches and resumes only when the model, corpus, dimensions, and
missing-row layout match. Query metrics checkpoint every 100 queries by default. Completed result
files are in [`experiments/results/`](../experiments/results/), and the frozen construction policy
is in [`configs/corpus_repair.toml`](../configs/corpus_repair.toml).
