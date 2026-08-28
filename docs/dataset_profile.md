# SG-LegalCite Phase 0 profile

Observed on 24 August 2026 from pinned Hugging Face revision
`3b25b337bcd2e0e7547570f170028900c931ebbd`.

## Core CSV

| Measure | Observed |
|---|---:|
| Rows | 100,890 |
| Rows eligible for every query mode | 100,884 |
| Unique citing judgments | 8,523 |
| Unique principle strings | 72,500 |
| Unique cited-case strings | 48,478 |
| Unique issues | 86,519 |
| Unique issue groups | 9,748 |
| Year range | 2000–2025 |
| Duplicate `(judgment, cited case, principle)` labels | 43 |
| Suspiciously long cited-case values | 584 |

The headline row, judgment, principle, cited-case, issue, and issue-group counts exactly match the
upstream dataset card. Six rows have empty principle, issue, and issue-group values. Four belong to
`[2002] SGHC 207`; the others belong to `[2006] SGHC 143` and `[2006] SGHC 162`. Those six rows are
excluded from the common query-mode evaluation subset so all retrieval ablations use identical
labels.

The 584 suspicious targets are a triage heuristic: a cited-case value over 500 characters, or over
180 characters without a Singapore neutral citation. This is not proof that every flagged value is
wrong. Phase 1 must manually sample these records and build a canonical case-ID policy before using
exact string identity as retrieval ground truth.

## Primary temporal split

| Split | Years | Judgments | Rows |
|---|---:|---:|---:|
| Train | 2000–2021 | 6,950 | 78,659 |
| Validation | 2022–2023 | 833 | 11,240 |
| Test | 2024–2025 | 740 | 10,991 |

There is no `Judgment_URL` overlap. The heuristic audit finds 85 normalized citing-case families and
5,234 cited targets in more than one split. In a forward-looking temporal evaluation, earlier
related litigation and a shared precedent corpus are realistic information, so these are reported
rather than removed. No claim is made that normalized names identify every related proceeding.

## Judgment-grouped random comparison split

| Split | Judgments | Rows |
|---|---:|---:|
| Train | 6,818 | 80,962 |
| Validation | 852 | 9,871 |
| Test | 853 | 10,057 |

This seeded 80/10/10 split has no judgment URL overlap but has 288 heuristic case families crossing
splits. It exists for a conventional random-split comparison and is not the primary production
estimate.

## Authors' 1,000-way benchmark artifacts

| Measure | Fact-only pool | Principle-augmented pool |
|---|---:|---:|
| Queries | 9,942 | 9,979 |
| Uniform 1,000-way pools | Yes | Yes |
| Pools with duplicate IDs | 0 | 0 |
| Gold IDs absent from pool | 0 | 0 |
| IDs absent from lookup | 0 | 0 |
| Gold-name/lookup mismatches | 0 | 0 |

The case lookup has 48,298 IDs, 180 fewer than the CSV's unique cited-case strings. The two pool
files contain the same 9,942 unique `(fact, cited case)` pairs, while the principle pool contains 37
additional triples where a fact/target pair has another principle. Numeric pool IDs are not stable
semantic keys: only 80 same-numbered records contain the same fact/target pair. A paired query-mode
ablation must therefore rescore the same principle candidate pool with and without its principle;
joining the two releases on numeric pool ID would silently compare unrelated examples.

The authors' sampled-pool results and this project's future full-corpus results answer different
questions. They will be reported in separate tables and never compared as if candidate difficulty
were equal.
