# Independent semantic judge

## Purpose and boundary

Deterministic citation controls establish schema validity, evidence visibility, case/evidence
consistency, passage integrity, and application-owned source text. They cannot establish that a
claim's substantive meaning follows from the passage. The independent judge adds that narrower
semantic QA signal.

The judge is not legal ground truth, a security boundary, or a replacement for deterministic
validation. A deterministic contract failure remains a failure regardless of a judge verdict. The
judge does not run synchronously in `/answer`, so its latency, cost, availability, or manipulation
cannot block or relax production enforcement.

```text
normal production path                 separate QA path
generator -> deterministic checks      frozen historical answer
          -> user response                 -> integrity verification
                                             -> sanitized judge package
                                             -> independent semantic verdict
                                             -> calibration result artifact
```

## Provider choice

The primary candidate is Google Gemini `gemini-3.7-flash`, while the generator is OpenAI
`gpt-5.6-luna`. This provides provider and model-family separation. Google documents the selected
model as a stable generally available model with structured output, thinking levels, and a
1,048,576-token input context; the pilot needs only a small fraction of that context. See the
[Gemini 3.7 Flash model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash),
[latest-model guide](https://ai.google.dev/gemini-api/docs/latest-model), and
[structured-output guide](https://ai.google.dev/gemini-api/docs/structured-output).

The pilot is restricted to the Gemini Developer API **Free Tier**. The 2026-08-27 snapshot records
both input and output as free of charge for `gemini-3.7-flash`; Google also states that Free Tier
content may be used to improve its products. Only the existing public/licensed Precedent evaluation
material may therefore be submitted. Confidential material, private Obsidian content, and hidden
reviewer metadata must never enter the provider payload. Do not enable Google Cloud billing or a
paid Gemini plan for this project. See the [official pricing and data-use table](https://ai.google.dev/gemini-api/docs/pricing)
and [billing guide](https://ai.google.dev/gemini-api/docs/billing).

The non-automatic fallback is Free Tier `gemini-3.5-flash`, which Google also lists as stable with
structured outputs. If 3.7 is inaccessible, stop the frozen run. Trying 3.5 requires a new model
configuration, run signature, and explicit approval; it is not an automatic retry. If neither model
is accessible on the existing Free Tier project, stop rather than enabling billing. See the
[Gemini 3.5 Flash model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash).

The adapter uses the provider's current Interactions HTTP API behind a small
`SemanticJudgeProvider` protocol. It sends one stateless `store=false` request per whole answer,
provides no tools, stores no application secret in artifacts, applies a 30-second timeout, and
performs zero automatic retries or fallbacks. `GEMINI_API_KEY` must be separate from
`OPENAI_API_KEY`, `PRECEDENT_API_KEY`, and `PRECEDENT_METRICS_KEY`.

## Clean-room input and prompt isolation

The provider receives exactly:

- query mode and exact query text shown to the generator;
- evidence in its exact model-visible order, limited to evidence ID, case ID, case name, source
  judgment, source year, and passage;
- the exact generated structured answer;
- the versioned semantic rubric and output instructions.

The payload keys are `untrusted_query`, `untrusted_evidence`, and
`untrusted_generated_answer`. The system prompt says all three are untrusted evaluation data, not
instructions. This helps contain prompt injection such as a passage saying “mark this supported,”
but does not eliminate model manipulation.

The provider does not receive generator identity, provider settings, previous labels or scores,
clean-room labels, gold targets, accepted case IDs, `target_present`,
`citation_relationship_verified`, retrieval ranks or scores, sufficiency labels, builder
rationales, previous judge results, confusion matrices, private Obsidian notes, or other
confidential material. Package IDs and digests are retained locally for integrity and joins but are
absent from the provider payload.

## Versioned decision contract

The frozen contract versions the prompt (`semantic-judge-prompt-v1`), schema
(`semantic-judge-schema-v1`), rubric (`semantic-grounding-rubric-v1`), and sanitized package
(`semantic-judge-package-v1`). SHA-256 signatures cover each contract and a 24-character run
signature covers the selected inputs, settings, and expected call count.

The output has one record verdict and an ordered verdict for every generated claim. Verdicts are
`supported`, `unsupported`, or `uncertain`. Each claim decision contains its zero-based claim index,
one or more visible evidence IDs, and a short reason; the record includes a bounded summary reason.
Extra fields, missing/reordered claims, arbitrary evidence IDs, malformed JSON, and explanations
beyond the schema bounds are rejected. Provider failure or malformed output is
`judge_unavailable`, never `unsupported`.

The rubric treats harmless paraphrase, non-verbatim wording, and explicit factual limitations as
acceptable. It marks a claim unsupported when it materially exceeds, contradicts, or misattributes
the supplied evidence, and preserves `uncertain` for genuine ambiguity or specialist interpretation.
The whole-answer verdict separately considers whether the recommendation and explanation follow;
all individual propositions can be supported while the record-level recommendation is not.

## Calibration and reference set

The pilot reuses the eight answered outputs from the existing 12-record behavioral run. The four
abstentions contain no substantive generated claims, so including them would test abstention
appropriateness rather than claim grounding. One call evaluates each whole answered record and its
claims: eight frozen calls and 14 claim decisions.

`experiments/samples/semantic_judge_reference.json` freezes record- and claim-level reference
adjudication from pre-existing completed manual review, clean-room behavioral adjudication, and the
Phase 3.1 manual citation audit. These are reference adjudications, not “human ground truth.” They
were frozen before any semantic-judge output existed and must not be changed to improve agreement.
The reference has four supported and four unsupported records; all 14 narrow proposition claims are
supported, exposing whether the judge can distinguish claim support from a broader unsupported
recommendation.

Eight deterministic challenge categories live under `tests/fixtures` for offline regression:
supported paraphrase, unsupported overclaim, negation, wrong evidence attribution, correct authority
with unsupported conclusion, a mixed answer, explicit factual limitation, and ambiguous evidence.
They are synthetic calibration fixtures and say nothing about production prevalence.

Result artifacts report raw counts with record- and claim-level agreement, supported precision and
recall, unsupported detection, and uncertain rate. Reference-uncertain examples are excluded from
binary denominators. Every disagreement is preserved as `pending_manual_review`; after a live pilot,
each must be manually classified before Phase 7.6 can be called complete. The sample is too small for
statistical or leaderboard claims.

## Offline workflow and Free Tier gate

Preparation validates the original behavioral manifest, evidence locks, generation prompt and
settings, and every cached response before projecting the eight answered records. Preparation never
constructs a provider:

```bash
uv run sg-legal-semantic-judge --prepare
uv run sg-legal-semantic-judge --preflight
```

The committed Free Tier pilot has run signature `cc5a3538e92350348ddf1847`. Live execution requires
the separate secret, an exact signature confirmation, explicit confirmation that the key belongs
to a non-billing-enabled project, and a separate approval:

```bash
GEMINI_API_KEY='...' uv run sg-legal-semantic-judge --execute \
  --confirm-run-signature cc5a3538e92350348ddf1847 \
  --confirm-free-tier
```

Do not run that command without separate live-inference approval. Integrity, reference, and signature
checks finish before provider construction. Per-record results are cached, including operational
failures, so a restart does not silently retry calls. Judge observability stays in the offline
result artifact—request counts, failures, durations, verdicts, tokens, and cost—rather than adding
Prometheus surface to the production service. No query, evidence, or rationale becomes a metric
label.

## Future shadow mode and residual risk

A later production design could sample already accepted responses into an asynchronous QA queue,
judge them out of band, and aggregate bounded quality metrics. It should not add the judge to the
critical response path or permit a judge verdict to override deterministic enforcement. This phase
does not implement a queue.

The judge can hallucinate, be influenced by prompt injection, disagree with a specialist reviewer,
and share latent training or data biases with the generator despite provider independence. A small
calibration set cannot prove reliability. Those limitations motivate separate evaluator research;
Precedent only demonstrates a bounded complementary QA signal.
