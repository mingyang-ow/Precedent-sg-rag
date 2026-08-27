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

The oracle, abstention, and prompt-sufficiency methodology is now repaired offline. The repaired
`rag-v2` canary answered correctly on the same Ahmed Salim evidence. The ten retrieved packages in
the frozen 12-case pilot were then manually adjudicated, blind to their model outputs. No API calls
were made during adjudication. That pilot is preserved unchanged as an answer-generation
diagnostic. A separate balanced behavioral pilot has now been frozen after deterministic blind
sequential evidence review. It contains six answer-expected and six abstain-expected records,
balanced 6/6 across facts-only and facts-plus-principle queries. The current decision is:

> **BALANCED BEHAVIORAL PILOT FROZEN — PAID INFERENCE APPROVAL HOLD**

All ten retrieved packages were judged answerable, so all 12 pilot records now expect an answer.
The separate behavioral sample measures both sides of the decision boundary. Paid inference
remains unapproved, and no API calls were made while constructing either ground truth.

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
output. Three later canaries each made exactly one successful request, including the successful
`rag-v2` Ahmed Salim canary. The blind adjudication made no API requests.

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

The schema-v4 manifest at `experiments/samples/rag_baseline.json` preserves query IDs, strata,
conditions, depths, the run signature, cost forecast, evidence origin and gold-row provenance,
expected-action labels, ordered passage digests, per-package model-visible evidence signatures,
exact rendered-input signatures, and one global evidence signature. The input-only
`experiments/samples/rag_behaviour_packages.json` artifact separately preserves the 12 complete,
ordered package payloads needed for behavioral-pilot inference. It contains no provider outputs.
The fast execution path validates those payloads against the manifest locks and signatures before
provider construction; full retrieval reconstruction remains a separate reproducibility audit.
Full prompts, evidence, outputs, provider metadata, latency, token use, and estimated cost are
cached per record under ignored `data/processed/generation/` paths.

Prompt `rag-v2` has signature
`29fa06887d945fd91959c89b6d9637d0cb732beb21ae4f5d2bd001aa9e3446be`. The blind pilot
adjudication is frozen separately as `pilot-sufficiency-v1` with digest
`1d603177732e892f150cdffead63e85242b2b865b8e8dcc22dd56182dbc4fd03`. Prompt text, versioned
settings, evidence, and the ground-truth digest are part of the cache identity, producing run
signature `12498ed3148d7ae999e76150`. This new identity prevents the prior canary cache from reducing
a later approved pilot below 12 fresh calls. The global evidence signature remains
`39d7ce7a0e8a0164712b4dbf1b4fa042b49222c1b6f409f800d0e95805cd29fe`.

The 96 sampled query IDs and their strata are byte-for-byte unchanged from methodology v1. Of the
352 expected actions, 290 changed: 76 retrieved `answer` labels and 212 retrieved `abstain` labels
became `unknown_needs_review`, as did two oracle rows whose labelled citation relationship could
not be verified in the supplied paragraph. The other 62 oracle rows are verified answer cases. The
ignored `data/processed/rag_sufficiency_review.json` file contains the private review queue.

The separate, version-controlled `experiments/samples/rag_pilot_adjudication.json` artifact covers
only the ten retrieved pilot packages and was created from query, evidence, and source metadata
without reading their model outputs. It records the target-presence diagnostic, manual sufficiency,
expected action, rationale, cited evidence passages, support granularity, reviewer, date, version,
and borderline status. All ten are `answer`; four had target present and six did not. Two are
borderline: `12e71dc2c2cb5a60f8075814` is useful mainly as a factual/sentencing comparator, and
`02bba523326c5266cf09e44e` supplies the Ladd test and attenuation principle but not the
committal-specific power. The other eight were sufficiently direct. The resulting full-pilot
ground truth is 12 answer and zero abstain.

The answer-only artifact is not overwritten or repurposed. The separate
`experiments/samples/rag_behaviour_adjudication.json` audit freezes the deterministic seed, all 278
candidate IDs in their pre-inspection hash order, all 52 retrieved candidates reviewed before the
balanced quotas filled, and three separately reviewed oracle candidates. Every inspected package
is recorded, including those not selected. The compact
`experiments/samples/rag_behaviour_pilot.json` manifest freezes the selected 12 IDs, evidence and
adjudication digests, unchanged `rag-v2` generation contract, estimate, and a distinct cache/run
signature. Model outputs were not inspected during selection or adjudication.

The behavioral pilot is intentionally 50% answer and 50% abstain. It is a decision-boundary
diagnostic, not an estimate of natural production class prevalence.

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
| Balanced behavioral pilot | 12 | 22,833 | 2,160 | 7,200 | $0.0072 | $0.0132 |
| Full baseline | 352 | 777,428 | 63,360 | 211,200 | $0.2315 | $0.4089 |

The original 12-call answer-only pilot contains six records per query mode: an answer-expected oracle warm-success
record, target-present retrieved records at depths 1 and 5, a warm retrieval-failure at depth 5,
and cold-start evidence at depths 1 and 5. Blind manual review found all five retrieved records per
mode sufficient for a bounded precedent answer. Target presence still does not determine their
expected action: six of the ten retrieved packages lack the labelled target but contain other
useful supplied authority.

The separate behavioral pilot contains one manually verified oracle answer, two retrieved answers,
and three retrieved abstentions per query mode. Its retrieved top-k distribution is four at k=1,
five at k=3, and one at k=5. Its frozen adjudication digest is
`5bfca978eef01713f937a08f9212a79aa01a928fb2b0d17ee89f687ce0ba9a15`, selected-evidence
digest is `9faf464cb462aa3a4b87a13942f7bea4f7c81cba6db99a97ea8a165aca5cebb5`, and cache/run
signature is `3664b44b7d4dbe620225d598`.

That 6/6 adjudication is retained as the original historical evaluation layer. A later methodology
audit found that its reviewer had access to citation-verification and retrieval diagnostics that
were not visible to the model. A context-isolated reviewer therefore re-adjudicated the same 12
unchanged inputs using only the exact query, ordered prompt evidence, visible case/source fields,
and `rag-v2` answerability rule. The parent agent recused itself because it had already seen prior
results and rationales.

The new `behaviour-cleanroom-v1` layer is frozen in
`experiments/samples/rag_behaviour_cleanroom_adjudication.json`, digest
`f3915f8202a56f687cc85655290532d36da6603c7c636e57b8500a90845eb6db`. It labels nine records
answer and three abstain, with no forced borderline or cannot-determine labels. Five labels differ
from the original evaluation: four hidden-metadata abstentions become answers based on useful
visible propositions, while one oracle answer becomes an abstention because its visible passage
does not develop a useful rule for the query.

Against the preserved outputs, the clean-room confusion matrix is TP 7, TN 2, FP 1, FN 2. Answer
recall is 0.778, abstention recall is 0.667, balanced accuracy is 0.722, false-answer rate is 0.333,
and false-abstention rate is 0.222. Oracle records are 2/2; retrieved records have 0.75 answer
recall, 0.50 abstention recall, and 0.625 balanced accuracy. Facts-only balanced accuracy is 0.667;
facts-plus-principle has 0.833 answer recall but no abstain-labelled denominator.

All 12 provider/schema calls succeeded. Eight outputs answered. Under the unchanged strict
verbatim-quote evaluator, three of eight answered records are fully citation-valid, mean citation
correctness is 0.4375, citation completeness is 1.0, unsupported-claim rate is 0.5625, and no
output cites an unsupplied authority. Four of fourteen claims have a deterministic mojibake
equivalence; counting that evaluator-only equivalence would raise valid claims from 7 to 11 and
fully valid answered records from 3 to 5. It is reported separately rather than replacing the
primary metric because `rag-v2` explicitly requires verbatim quotes. Changing model-visible text
or exposing citation-verification metadata would require a new evidence digest, run signature, and
pilot and is not applied retroactively.

## Phase 3.1 citation evaluator audit

Strict citation matching remains the primary historical metric. In the current implementation,
"strict" already means substring matching after NFKC and whitespace canonicalization; Phase 3.1
does not change that behavior. A separate evaluator-only normalized mode reports the earliest
successful comparison stage—raw exact, NFKC, whitespace, observed mojibake equivalence, or no
match—and adds only two mappings justified by the frozen outputs:

| Corrupted evidence text | Canonical comparison text | Observed effect |
| --- | --- | --- |
| `â` | `–` | Three claims across two written-resolution packages |
| `â` | `’` | One sentencing-factor claim |

No generic encoding repair, fuzzy matching, semantic similarity, embedding, or model judge is
used. Both source strings remain immutable. The normalized metric is evaluation robustness, not
better model performance.

| Metric | Strict | Normalized | Absolute change |
| --- | ---: | ---: | ---: |
| Fully valid answered records | 3/8 | 5/8 | +2 |
| Citation validity | 0.375 | 0.625 | +0.250 |
| Mean citation correctness | 0.4375 | 0.7500 | +0.3125 |
| Citation completeness | 1.0000 | 1.0000 | 0 |
| Unsupported-claim proxy | 0.5625 | 0.2500 | -0.3125 |

Four of fourteen claims across three records change under normalized matching. Manual inspection
classified all four as evaluator artifacts: the quotations are otherwise contiguous and identical.
Three claims remain failures. Each uses literal ellipses to splice or omit passage words; each
underlying proposition is present in the cited passage, but the quoted text is not contiguous and
therefore remains a genuine `quote_not_found` citation-contract error. Citation-target validity,
quote validity, claim support, and whether the recommended authority was supplied are reported as
separate dimensions in the audit artifact.

This is a mixed result: deterministic encoding equivalence explains most failed claims (4/7) and
two failed records, but three genuine quotation failures remain. Changing evidence text or adding
verification metadata to the prompt would be a model-visible Type B change requiring a new evidence
digest, run signature, and pilot; Phase 3.1 makes no such change.

Automatic SDK retries are disabled (`max_retries=0`), so 352 logical calls mean 352 planned HTTP
attempts. Cache resumption makes completed records free to reuse. The command exits non-zero when
all provider attempts in an invocation fail. The explicit `--retry-errors` option can add at most
one later logical call for each failed cached record; it is never enabled by default.

## Important assumptions and limitations

Four limitations must not be mistaken for ground truth:

1. Oracle gold context is the citation passage from the test-query row, not the authoritative full
   text of the cited judgment. Two sampled rows do not textually verify their labelled citation
   relationship and remain `unknown_needs_review`.
2. Target presence and absence are retrieval identity diagnostics, not answerability labels.
   SG-LegalCite labels may be incomplete, and even the labelled case may be represented by an
   irrelevant historical proposition. Manual sufficiency review is required for retrieved context.
3. Exact-quote validation proves traceability, not that a claim is entailed by the quote. The manual
   semantic subset is part of the completion gate.
4. The preserved answer-only pilot has no abstain-expected record after blind adjudication. It can test answer
   generation, grounding, oracle-versus-retrieved behavior, and facts-only versus assisted queries,
   but it cannot estimate abstention correctness or inappropriate-answer rate. The separate
   balanced behavioral pilot measures that decision boundary, but its engineered class balance must
   not be interpreted as prevalence.

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
12-record answer-only pilot; and `--behaviour-pilot` restricts it to the separate frozen balanced
pilot. Behavioral-pilot execution loads only its 12 input-only packages, then verifies package
count and order, model-visible evidence and rendered-input signatures, prompt and output schema,
model and generation settings, global and selected evidence digests, both adjudication bindings,
and the run signature before provider construction. A safe dry preflight is:

```bash
uv run sg-legal-rag-evaluate --preflight-only --behaviour-pilot
```

The expensive end-to-end reproducibility audit remains available separately and never constructs
the provider:

```bash
uv run sg-legal-rag-evaluate --reconstruct-and-verify --behaviour-pilot
```

The clean-room review export and cached-output-only recomputation are separate no-provider paths:

```bash
uv run sg-legal-rag-cleanroom --export-review
uv run sg-legal-rag-cleanroom --evaluate
uv run sg-legal-rag-citation-audit
```

## Completion gate

Proceed to production engineering only if retrieved-context grounded end-to-end success and
abstention behavior are credible on both query modes, the oracle/retrieval gap is understood, the
manual semantic review finds no material unsupported-claim pattern, and cold-start limitations can
be surfaced honestly. The answer-only pilot cannot satisfy the abstention part of this gate on its
own. Until inference and an independently approved abstention evaluation are complete, the gate
remains closed.
