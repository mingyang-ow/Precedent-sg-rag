# Production citation contract

## Status

Phase 4 established the application boundary for evidence-traceable citations. It did not call a
model, change retrieval, or reinterpret the Phase 3 experiments. Phase 5 now exposes the boundary
through the typed [FastAPI service](api.md).

Earlier bounded-RAG evaluation required the model to copy a verbatim quotation. Phase 3.1 showed
why that is a poor production responsibility: four apparent failures were encoding artifacts, and
the remaining three were supported propositions expressed with non-contiguous ellipses. Those
historical strict and normalized results remain preserved. The production contract instead makes
the model identify evidence while the application owns authoritative text.

## Architecture

The historical evaluation path remains:

```text
GroundedAnswer / rag-v2
    -> model-generated statement + evidence_id + supporting_quote
    -> strict and normalized quotation evaluation
```

The production path is:

```text
ProductionAnswer / production-citation-v1
    -> model-generated statement + evidence_id + case_id
    -> validate against the exact model-visible evidence set
    -> verify the immutable passage digest
    -> resolve exact passage and source metadata from the frozen evidence store
    -> ResolvedProductionAnswer
```

`source_text` is deliberately absent from `ProductionClaim`. Extra model fields are rejected, so
the model cannot override the passage displayed by the application.

## Version boundary

| Contract | Version/signature | Purpose |
| --- | --- | --- |
| Historical prompt | `rag-v2` / `29fa0688...3446be` | Frozen Phase 3 evaluation |
| Historical schema | `GroundedAnswer` / `61e54fb6...87eeb` | Generated verbatim quotations |
| Production prompt | `rag-production-v2` / `b4aacdd6...064a3e` | Untrusted query/evidence envelope; evidence references only |
| Production schema | `production-citation-v1` / `4ca7a25a...be65` | Application-resolved citations |

The production prompt preserves the `rag-v2` distinction between identifying relevant authority
and proving that the client's facts satisfy the rule. Relevant authority with unresolved factual
application still receives an answer with a limitation; absent, unrelated, ambiguous, or too-weak
evidence requires abstention.

Phase 7.5 revised only the never-executed production prompt. It explicitly treats the user query
and every evidence field as untrusted data that cannot supply instructions, authorize tools,
change the output schema, or request secrets or internal configuration. The deterministic provider
input nests the query and ordered evidence beneath `untrusted_data` in JSON, so malicious JSON-like
text remains a string value. This is defense-in-depth, not a claim that prompting prevents every
semantic injection.

No production inference artifact or run signature exists yet. A future model execution must freeze
the new prompt, schema, evidence, settings, and evaluation labels under a new run signature. It
must not reuse the historical Phase 3 identity.

## Model and resolved objects

The model-facing object contains references, not source prose:

```json
{
  "contract_version": "production-citation-v1",
  "status": "answered",
  "recommended_case_id": "case:941",
  "explanation": "The authority states the test, but application remains unresolved.",
  "claims": [
    {
      "statement": "The authority identifies three cumulative requirements.",
      "evidence_id": "E1",
      "case_id": "case:941"
    }
  ]
}
```

Resolution returns the exact stored passage and provenance:

```json
{
  "statement": "The authority identifies three cumulative requirements.",
  "citation": {
    "evidence_id": "E1",
    "case_id": "case:941",
    "case_name": "Ahmed Salim v Public Prosecutor",
    "source_text": "Exact stored passage...",
    "source_judgment": "[2024] SGCA 1",
    "source_year": 2024,
    "passage_digest": "...",
    "retrieval_rank": 1,
    "retrieval_score": 12.5
  }
}
```

Evidence IDs are package-local because the existing retrieval contract uses `E1`, `E2`, and so
on. A resolved response therefore also carries `package_id` and `query_id`; the passage digest
provides content integrity. This avoids pretending that a local rank identifier is globally
unique.

## Deterministic validation

The resolver derives the visible ID allow-list from the same `prompt_evidence` projection used for
model input. It rejects, without repair:

- malformed or extra structured-output fields;
- an unknown evidence ID;
- a known stored ID that was not visible to the model;
- a claim whose case ID disagrees with its evidence item;
- an unsupplied recommended case;
- duplicate evidence references;
- an answered response with no recommendation or supporting claim;
- an abstention with a recommendation or claim citation;
- a stored passage whose SHA-256 digest no longer matches.

Only after every check passes does the resolver copy `passage`, case metadata, judgment metadata,
year, URL, retrieval rank, and retrieval score into the resolved response. The raw model object and
frozen evidence package are immutable throughout resolution.

This improves reliability by removing brittle copying, traceability by retaining a deterministic
claim-to-passage chain, auditability through explicit identifiers and digests, user trust by
displaying application-owned sources, and production simplicity by replacing quote comparison
with referential-integrity checks.

## Deliberate limits

The resolver proves provenance, not semantic entailment. Claim-support evaluation remains a
separate concern; Phase 4 does not introduce an LLM judge, embeddings, or fuzzy matching. The
frozen `EvidencePackage` is the current storage boundary rather than a database-backed evidence
service. The API now adds service authentication and resource controls around this boundary.
Semantic claim-support evaluation remains Phase 7.6 work and does not replace the deterministic
resolver.
