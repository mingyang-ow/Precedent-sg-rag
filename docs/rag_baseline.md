# Bounded grounded RAG baseline

## Status

Phase 3 is implemented and verified offline. The deterministic generation subset is frozen, but no
model inference has been run. The current decision is therefore:

> **RAG GENERATION NOT YET RELIABLE — PRE-INFERENCE HOLD**

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

This would be separately billed OpenAI API usage. It would not consume a ChatGPT subscription or a
Codex interactive allowance. No API credential was configured during offline preparation, and no
inference request was issued.

## Frozen experiment

The selection seed is `20260826`. Sampling is balanced over two query modes and three retrieval
strata, with 16 underlying queries in every mode/stratum cell.

| Dimension | Values | Records |
| --- | --- | ---: |
| Query mode | facts only; facts plus principle | 176 each |
| Underlying-query stratum | warm retrieval success; warm retrieval failure; cold start | 32 each |
| Oracle condition | one known-relevant historical context for every warm query | 64 |
| Retrieved sufficient condition | frozen passage BM25 at case depths 1, 3, and 5 | 76 |
| Insufficient-evidence condition | target absent from the retrieved case set | 212 |
| Total | 96 underlying queries expanded across conditions/depths | 352 |

Every one of the 96 underlying queries receives retrieved evidence at case depths 1, 3, and 5.
Every warm query also receives one oracle-context run. BM25 ranks positive lexical matches only,
aggregates passages to cases by maximum score, and packages the best passage for each selected
case. Arbitrary zero-score cases are never sent to generation.

Facts-only is the primary product-facing diagnostic. Facts-plus-principle is retained as an
assisted-query comparison, not as evidence available to a real user at prediction time.

The committed manifest at `experiments/samples/rag_baseline.json` preserves query IDs, package IDs,
strata, conditions, depths, the run signature, and cost forecast without redistributing prompt
passages. Full prompts, evidence, outputs, provider metadata, latency, token use, and estimated cost
are cached per record under ignored `data/processed/generation/` paths.

## Prompt and output contract

The prompt allows only supplied evidence. It prohibits outside legal knowledge, invented cases,
sources, propositions, and quotations. An answer must select a supplied `case_id`, express at most
four atomic claims, cite the supplied `evidence_id` for each claim, and copy a short verbatim quote.
If the evidence is absent or too weak, it must return the fixed explicit abstention.

The typed output has four fields:

- `status`: `answered` or `insufficient_evidence`;
- `recommended_case_id`: a supplied case ID or null;
- `explanation`: a bounded explanation;
- `claims`: atomic statements, each with an evidence ID and supporting quote.

Pydantic forbids extra fields and enforces status-dependent invariants. Deterministic post-parse
checks reject unsupplied recommendations, unknown evidence IDs, non-verbatim quotes, unseen case
IDs, and unseen Singapore neutral/report citations.

## Evaluation

Automated metrics are computed before any model-based or manual judgment:

- citation correctness from valid evidence identifiers and exact passage quotes;
- citation completeness from the required citation on each structured claim;
- unsupported-claim proxy from invalid quotes or unseen identifiers/citations;
- labelled-precedent correctness;
- abstention precision, recall, and inappropriate-answer rate;
- grounded generation success;
- grounded end-to-end success, requiring retrieval success and a valid, fully cited, labelled-correct
  recommendation;
- latency, token use, estimated cost, query mode, top-k, and warm/cold breakdowns.

Exact quotation is not semantic entailment. A deterministic 36-record review template is therefore
created after inference, balanced over mode and condition. A human reviewer must label semantic
support, citation completeness, unsupported claims, and abstention appropriateness. An LLM judge is
not treated as ground truth.

The failure analysis keeps retrieval/generation and abstention views separate:

1. retrieval correct, generation correct;
2. retrieval correct, generation incorrect;
3. retrieval incorrect, answer grounded to the wrong supplied evidence;
4. retrieval incorrect, answer unsupported;
5. insufficient evidence, correct abstention;
6. insufficient evidence, inappropriate answer or provider/schema error.

A record can carry layer 3 or 4 in the retrieval/generation view and layer 6 in the abstention view.
This avoids hiding whether an inappropriate answer was at least grounded to what retrieval supplied.

## Request and cost forecast

Local `tiktoken` estimation includes instructions, the JSON schema, the exact prompt, and a fixed
request-overhead allowance. Server-side accounting can differ. Pricing is frozen to the official
model-page values observed on 26 August 2026: $0.20 per million input tokens, $0.02 per million
cached input tokens, and $1.20 per million output tokens.

| Run | Calls | Estimated input | Expected output | Configured output ceiling | Expected cost | Ceiling cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small pilot | 12 | 27,549 | 2,160 | 7,200 | $0.0081 | $0.0142 |
| Full baseline | 352 | 734,926 | 63,360 | 211,200 | $0.2230 | $0.4004 |

The 12-call pilot contains six records per query mode: oracle warm-success, retrieved-success at
depths 1 and 5, warm retrieval-failure at depth 5, and cold-start evidence at depths 1 and 5. It
covers all three conditions, both query modes, warm/cold behavior, and the top-k endpoints.

Automatic SDK retries are disabled (`max_retries=0`), so 352 logical calls mean 352 planned HTTP
attempts. Cache resumption makes completed records free to reuse. The explicit `--retry-errors`
option can add at most one later logical call for each failed cached record; it is never enabled by
default.

## Important assumptions and limitations

Three assumptions must not be mistaken for ground truth:

1. A known-relevant case's historical citation paragraph is only a bounded oracle proxy. It is not
   the authoritative full text of the cited judgment and may not express every fact or holding
   needed by the query.
2. “Insufficient evidence” is labelled when no accepted dataset target occurs in the retrieved set.
   SG-LegalCite labels may be incomplete, so a retrieved unlabelled case can still be relevant.
   Manual abstention review is required.
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

The billed path requires the explicit `--execute` flag. It must not be used until the request plan
has been approved. `--pilot` restricts that path to the fixed 12-record pilot.

## Completion gate

Proceed to production engineering only if retrieved-context grounded end-to-end success and
abstention behavior are credible on both query modes, the oracle/retrieval gap is understood, the
manual semantic review finds no material unsupported-claim pattern, and cold-start limitations can
be surfaced honestly. Until inference and that review are complete, the gate remains closed.
