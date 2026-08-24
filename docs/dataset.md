# Dataset validation and split policy

## Verified upstream state

Phase 0 pins Hugging Face revision `3b25b337bcd2e0e7547570f170028900c931ebbd`.
At that revision, the core file is `COMBINED_ALL_CASES_FINAL_V2.csv` encoded as Latin-1 with
13 columns. The release metadata reports 100,890 case–principle pairs from 8,523 citing
judgments spanning 2000–2025 and a CC BY 4.0 dataset licence.

The GitHub dataset subdirectory currently documents `train.jsonl`, `val.jsonl`, `test.jsonl`,
and `candidate_pool.jsonl`, but those files are not present in the live Hugging Face revision.
The repository therefore treats the live artifact API as authoritative and records the mismatch.

## Validation gates

The streaming validator checks:

- exact ordered schema and integer years;
- non-empty query, principle, judgment ID, and cited-target fields;
- published row and judgment counts;
- within-judgment metadata consistency;
- neutral-citation-to-URL conflicts;
- duplicate `(judgment, cited case, principle)` labels;
- suspiciously long cited-case values that may be extraction passages rather than identifiers.

Warnings and row-level eligibility samples are retained in a machine-readable report. Schema
errors and invalid years are hard failures. Rows missing a required retrieval field are counted as
ineligible for the common query-mode evaluation subset; `sg-legal-validate --strict` additionally
returns a non-zero status when such rows exist.

## Split policy

The primary evaluation split is chronological by the citing judgment:

| Split | Citing-judgment years | Purpose |
|---|---:|---|
| Train | 2000–2021 | Model fitting and retrieval configuration development |
| Validation | 2022–2023 | Model and hyperparameter selection |
| Test | 2024–2025 | Final forward-looking evaluation |

This better represents deployment under legal and language drift than a random split. A seeded
80/10/10 judgment-grouped random split is also available for comparability, but it is secondary
and must not be presented as the production estimate.

All rows from one `Judgment_URL` stay together. The audit additionally measures normalized case
name families and cited targets that cross split boundaries. Target overlap is expected in a
transductive retrieval task: test queries may target a precedent already present in the shared
candidate corpus. It must not be confused with query-label duplication.

The split manifest covers every judgment. Experiment loaders must intersect it with the validator's
common eligible subset so facts-only, principle-only, and combined-query runs see identical labels.

## Known limitations and risks

- Facts, principles, issues, and issue groups are LLM-derived; they are not gold legal analysis.
- Expert validation was performed on extraction quality, not every record.
- `Cited Case` is an unstructured string and may contain extraction noise or variants of one case.
- Exact target-string metrics will undercount aliases unless a canonical identifier layer is built.
- The source does not provide a reliable proceeding-family ID. Normalized case-name auditing can
  reveal obvious overlap but cannot prove that related appeals or consolidated matters are isolated.
- The authors' published metrics use sampled 1000-way pools. Full-corpus metrics are materially
  harder and should be reported separately, never compared as if the candidate sets were equal.
- The oracle `Key Principles Illustrated` is extracted from citation context. A user-facing system
  must evaluate facts-only and user/predicted-principle settings to avoid overstating cold-start
  performance.

The observed counts and quality findings for the pinned release are recorded in
[`dataset_profile.md`](dataset_profile.md).
