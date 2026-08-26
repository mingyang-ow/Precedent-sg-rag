# Bounded grounded RAG baseline

## Status

Phase 3 is implemented and verified offline. The first 12-call pilot attempt was rejected before
inference because the Responses API request used the obsolete top-level `verbosity` field. A later
one-call canary proved that the repaired provider, schema, parsing, and citation-validation path
works, but it also exposed a methodology defect: the old "oracle" supplied an unrelated historical
passage merely because it mentioned the labelled case. The model correctly abstained while the old
evaluator incorrectly expected an answer. After oracle repair, a second canary used manually
verified evidence that identified *Ahmed Salim v Public Prosecutor* as authority for the three-part
diminished-responsibility test. The model incorrectly abstained because the prompt did not clearly
separate precedent relevance from ultimate factual application.

The oracle, abstention, and prompt-sufficiency methodology is now repaired offline. No API calls
were made during this prompt repair. The current decision is:

> **PROMPT METHODOLOGY REPAIRED OFFLINE — ONE-CALL CANARY HOLD**

The hold is deliberate: paid inference requires explicit approval after reviewing the request and
cost forecast below. This document must be updated with measured results and a manual semantic
review before the production-engineering gate can be reconsidered.

## Scope and model choice

The baseline uses one model, `gpt-5.6-luna`, through the OpenAI Responses API. This is the current
cost-sensitive high-volume model in OpenAI's model guidance, and it supports Structured Outputs.
The Python client parses directly into a strict Pydantic schema. The requested alias is recorded in
configuration, while every cache record stores the actual `response.model` returned by the API.

- API guidance: <https://developers.openai.com/api/docs/guides/latest-model>
- model page: <https://developers.openai.com/api/docs/models/gpt-5.6-luna>
- Structured Outputs: <https://developers.openai.com/api/docs/guides/structured-outputs>

This is separately billed OpenAI API usage. It does not consume a ChatGPT subscription or a Codex
interactive allowance. The prior failed pilot attempt made 12 rejected API requests with no model
output. Two later canaries each made exactly one successful request. This repair made none.

## Frozen experiment

The selection seed is `20260826`. Sampling is balanced over two query modes and three retrieval
strata, with 16 underlying queries in every mode/stratum cell.

| Dimension | Values | Records |
| --- | --- | ---: |
| Query mode | facts only; facts plus principle | 176 each |
| Underlying-query stratum | warm retrieval success; warm retrieval failure; cold start | 32 each |
| Oracle condition | exact gold citation-row context for every warm-query oracle slot | 64 |
| Retrieved condition | frozen passage BM25 at case depths 1, 3, and 5 | 288 |
| Target present in retrieved cases | identity diagnostic only; not sufficiency ground truth | 76 |
| Target absent from retrieved cases | identity diagnostic only; not sufficiency ground truth | 212 |
| Total | 96 underlying queries expanded across conditions/depths | 352 |

Every one of the 96 underlying queries receives retrieved evidence at case depths 1, 3, and 5.
Every warm query also receives one `oracle_gold_context` run. Oracle evidence is now taken from the
exact gold citation row attached to the sampled test query. It is an evaluation-only upper bound
and never enters the retrieval index or deployed candidate corpus.

Retrieved evidence is unchanged: citation passages from judgments dated no later than 2023 are
ranked by passage BM25, aggregated to cases by maximum score, and represented by the best passage
for each selected case. Arbitrary zero-score cases are never sent to generation. All 288 rebuilt
retrieved prompt-evidence signatures exactly match the prior manifest.

Facts-only is the primary product-facing diagnostic. Facts-plus-principle is retained as an
assisted-query comparison, not as evidence available to a real user at prediction time.

The schema-v3 manifest at `experiments/samples/rag_baseline.json` preserves query IDs, strata,
conditions, depths, the run signature, and cost forecast without redistributing prompt passages.
It also freezes evidence origin and gold-row provenance, expected-action labels, ordered passage
digests, per-package model-visible evidence signatures, exact rendered-input signatures, and one
global evidence signature. Before constructing an API client, the execution path reconstructs and
compares the full frozen protocol and evidence lock.
Full prompts, evidence, outputs, provider metadata, latency, token use, and estimated cost are
cached per record under ignored `data/processed/generation/` paths.

Prompt `rag-v2` has signature
`29fa06887d945fd91959c89b6d9637d0cb732beb21ae4f5d2bd001aa9e3446be`, producing run
signature `b1ce0f7b4a99cc4e33f47a81`. The prompt text and version are part of the cache identity. The
global evidence signature remains
`39d7ce7a0e8a0164712b4dbf1b4fa042b49222c1b6f409f800d0e95805cd29fe`.

The 96 sampled query IDs and their strata are byte-for-byte unchanged from methodology v1. Of the
352 expected actions, 290 changed: 76 retrieved `answer` labels and 212 retrieved `abstain` labels
became `unknown_needs_review`, as did two oracle rows whose labelled citation relationship could
not be verified in the supplied paragraph. The other 62 oracle rows are verified answer cases. The
ignored `data/processed/rag_sufficiency_review.json` file contains the private review queue.

## Prompt and output contract

Prompt `rag-v2` allows only supplied evidence and prohibits outside legal knowledge, invented
cases, sources, propositions, and quotations. Answerability means that the passage supports
identifying a precedent as relevant authority for a legal principle, rule, or test; it does not
mean that the passage resolves whether the client's facts satisfy that test. When the passage
states a directly applicable test, the model should answer and disclose unresolved factual
application in `explanation`. Case identity alone remains insufficient, and absent, unrelated,
ambiguous, or too-weak evidence requires the fixed abstention.

An answer must select a supplied `case_id`, express at most four atomic claims, cite the supplied
`evidence_id` for each claim, and copy a short verbatim quote. It must not claim that the present
facts satisfy a test unless the passage supports that application.

The typed output has four fields:

- `status`: `answered` or `insufficient_evidence`;
- `recommended_case_id`: a supplied case ID or null;
- `explanation`: a bounded explanation;
- `claims`: atomic statements, each with an evidence ID and supporting quote.

Pydantic forbids extra fields and enforces status-dependent invariants. Deterministic post-parse
checks reject unsupplied recommendations, unknown evidence IDs, non-verbatim quotes, unseen case
IDs, and unseen Singapore neutral/report citations.

## Evaluation

Each package now carries three separate evaluation fields:

- `target_present`: whether an accepted case identity occurs in the supplied cases;
- `evidence_sufficient`: `true`, `false`, or unknown pending review;
- `expected_action`: `answer`, `abstain`, or `unknown_needs_review`.

Case identity never sets evidence sufficiency. Retrieved sufficiency requires manual review of
whether the supplied passage supports a defensible answer to that query. Unknown records are not
placed into answer/abstention correctness denominators. Provider/API and Structured Output failures
remain separate operational statuses and can never be counted as model abstentions.

Automated metrics are computed before any model-based or manual judgment:

- citation correctness from valid evidence identifiers and exact passage quotes;
- citation completeness from the required citation on each structured claim;
- unsupported-claim proxy from invalid quotes or unseen identifiers/citations;
- labelled-precedent correctness;
- abstention precision, recall, and inappropriate-answer rate on reviewed records only;
- grounded generation success;
- grounded end-to-end success, requiring retrieval success and a valid, fully cited, labelled-correct
  recommendation;
- latency, token use, estimated cost, query mode, top-k, and warm/cold breakdowns.

Exact quotation is not semantic entailment. A deterministic 36-record review template is therefore
created after inference, balanced over mode and condition. Its rubric separates precedent
relevance, proposition support, factual-application limitations, unsupported factual conclusions,
citation completeness, and abstention appropriateness. An LLM judge is not treated as ground truth.

The model-quality failure analysis keeps provider, retrieval identity, evidence sufficiency, and
generation behavior separate:

0. provider/API or Structured Output failure, recorded as distinct statuses;
1. target present, generation correct on reviewed-sufficient evidence;
2. target present, generation incorrect on reviewed-sufficient evidence;
3. target absent, answer grounded to the wrong supplied evidence;
4. target absent, answer unsupported;
5. reviewed-insufficient evidence, correct abstention;
6. reviewed-insufficient evidence, inappropriate model answer;
7. evidence sufficiency unknown and awaiting review.

A record can carry layer 3 or 4 in the retrieval/generation view and layer 6 in the abstention view.
This avoids hiding whether an inappropriate answer was at least grounded to what retrieval supplied.

## Request and cost forecast

Local `tiktoken` estimation includes instructions, the JSON schema, the exact prompt, and a fixed
request-overhead allowance. Server-side accounting can differ. Pricing is frozen to the official
model-page values observed on 26 August 2026: $0.20 per million input tokens, $0.02 per million
cached input tokens, and $1.20 per million output tokens.

| Run | Calls | Estimated input | Expected output | Configured output ceiling | Expected cost | Ceiling cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small pilot | 12 | 28,695 | 2,160 | 7,200 | $0.0083 | $0.0144 |
| Full baseline | 352 | 777,428 | 63,360 | 211,200 | $0.2315 | $0.4089 |

The 12-call pilot contains six records per query mode: an answer-expected oracle warm-success
record, target-present retrieved records at depths 1 and 5, a warm retrieval-failure at depth 5,
and cold-start evidence at depths 1 and 5. The five retrieved records per mode remain pending
sufficiency review; target presence alone does not determine their expected action.

Automatic SDK retries are disabled (`max_retries=0`), so 352 logical calls mean 352 planned HTTP
attempts. Cache resumption makes completed records free to reuse. The command exits non-zero when
all provider attempts in an invocation fail. The explicit `--retry-errors` option can add at most
one later logical call for each failed cached record; it is never enabled by default.

## Important assumptions and limitations

Three limitations must not be mistaken for ground truth:

1. Oracle gold context is the citation passage from the test-query row, not the authoritative full
   text of the cited judgment. Two sampled rows do not textually verify their labelled citation
   relationship and remain `unknown_needs_review`.
2. Target presence and absence are retrieval identity diagnostics, not answerability labels.
   SG-LegalCite labels may be incomplete, and even the labelled case may be represented by an
   irrelevant historical proposition. Manual sufficiency review is required for retrieved context.
3. Exact-quote validation proves traceability, not that a claim is entailed by the quote. The manual
   semantic subset is part of the completion gate.

The API does not expose a seed in this implementation, and temperature is omitted. Sample and
evidence selection are deterministic; model wording is not guaranteed byte-for-byte repeatable.

## Safe commands

Offline preparation is the default and cannot call the API:

```bash
uv sync --extra dev --extra generation
uv run sg-legal-rag-evaluate
```

The billed path requires the explicit `--execute` flag. It must not be used until a new request plan
has been approved. `--canary` restricts that path to the manually verified Ahmed Salim
`oracle_gold_context` package `0362866e90548d293669f8d3`; `--pilot` restricts it to the fixed
12-record pilot. No inference is authorized by this repair.

## Completion gate

Proceed to production engineering only if retrieved-context grounded end-to-end success and
abstention behavior are credible on both query modes, the oracle/retrieval gap is understood, the
manual semantic review finds no material unsupported-claim pattern, and cold-start limitations can
be surfaced honestly. Until inference and that review are complete, the gate remains closed.
