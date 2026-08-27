# FastAPI service

## Service boundary

Phase 5 exposes the production citation contract through a deliberately small HTTP surface:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness only |
| `GET` | `/ready` | Local retrieval and generation-configuration readiness |
| `GET` | `/version` | API, prompt, schema, and citation-contract identity |
| `POST` | `/retrieve` | Passage-BM25 evidence retrieval |
| `POST` | `/answer` | Retrieval, generation, citation validation, and source resolution |

Interactive OpenAPI documentation is available at `/docs`. The service is not legal advice.

## Local startup

```bash
uv sync --extra dev --extra generation --extra api
uv run uvicorn sg_legal_rag.api.app:app --reload
```

Then inspect:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/docs
```

Service startup, `/health`, and `/ready` never call a model provider. `/answer` is unavailable when
generation is not configured.

## Configuration

Configuration comes directly from environment variables; a `.env` file is not required.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Enables `/answer`; never returned or logged |
| `MODEL_ID` | `gpt-5.6-luna` | Future production generation model |
| `TOP_K_DEFAULT` | `5` | Retrieval depth when omitted |
| `MAX_TOP_K` | `10` | Public retrieval-depth ceiling |
| `MAX_OUTPUT_TOKENS` | `600` | Provider output ceiling |
| `REASONING_EFFORT` | `none` | Provider reasoning setting |
| `VERBOSITY` | `low` | Provider output verbosity |
| `PRECEDENT_DATA_DIR` | `data/raw` | SG-LegalCite source directory |
| `PRECEDENT_SPLITS_PATH` | `data/processed/splits_temporal.csv` | Temporal split artifact |
| `PRECEDENT_CORPUS_CONFIG` | `configs/corpus_repair.toml` | Leakage-safe corpus settings |

Readiness uses a partial policy. If local retrieval assets and configuration are valid but
`OPENAI_API_KEY` is absent, `/ready` returns HTTP 200 with `status: partial`: `/retrieve` is ready
and `/answer` returns 503. Missing retrieval assets produce HTTP 503 with `status: not_ready`.

## Retrieval and answer flow

`POST /retrieve` accepts bounded facts, an optional principle, and an optional `top_k`. The service
canonicalizes the query, lazily loads the leakage-safe historical passage corpus and BM25 index,
and returns exact application-owned `EvidenceItem` text and provenance. It does not expose
retriever tuning knobs.

`POST /answer` uses the same retrieval path, then invokes an injected
`ProductionGenerationProvider`, validates `production-citation-v1`, and resolves the source through
`FrozenEvidenceResolver`:

```text
HTTP request
    -> application service
    -> passage BM25
    -> bounded EvidencePackage
    -> production provider abstraction
    -> evidence visibility / case / status / digest validation
    -> application-owned source text
    -> typed AnswerResponse
```

The default OpenAI adapter is lazy. Even when credentials exist, constructing the application does
not construct an OpenAI client. Client construction and the provider request occur only inside an
explicit `/answer` operation. Tests inject a fake provider and require no network access.

The retrieval adapter replaces the historical full-paragraph digest with a SHA-256 digest of the
exact bounded passage displayed by the production application. Historical frozen evidence and
signatures are not changed.

## Errors

Errors use a typed body containing a stable code, safe message, request ID, and optional issue
codes. Stack traces, raw provider errors, query text, evidence text, and credentials are not
returned.

| HTTP | Category | Examples |
| ---: | --- | --- |
| 400 | Request policy | `top_k` outside configured bounds |
| 422 | Request schema | Blank facts, wrong types, unknown fields |
| 503 | Dependency unavailable | Missing retrieval assets or generation configuration |
| 502 | Provider/generated output | Provider failure, malformed output, hallucinated or mismatched citation reference |
| 500 | Integrity/internal | Changed passage digest or impossible retrieved-evidence invariant |

Unknown evidence IDs, evidence/case mismatches, hidden evidence references, answers without claims,
and abstentions with citations are treated as invalid generated output (`502`). A passage-digest
mismatch means application-owned evidence changed and is therefore an internal integrity failure
(`500`). Nothing is silently repaired.

## Request identity and logging

The service accepts a safe `X-Request-ID` containing up to 64 ASCII letters, digits, dots,
underscores, or hyphens. Otherwise it generates a UUID. Every HTTP response returns the ID in the
same header, and structured error bodies include it.

JSON application logs contain request ID, endpoint, method, HTTP status, total latency, retrieval
count, answer status, provider status, and retrieval/generation/resolution timings where relevant.
They deliberately omit full query and evidence text, API keys, and credential values. A formal
metrics backend is deferred to the observability phase.

## Bounded technical debt

Evidence IDs remain local to one package (`E1`, `E2`, ...). Responses therefore retain the
top-level `package_id`, while citations include `evidence_id` and `passage_digest`. Persistent
evidence storage is deferred to Phase 6.

The current production retriever lazily rebuilds the in-memory historical passage corpus and BM25
index from local assets on the first request. Phase 6 should persist or cache that initialized
state for faster cold starts. The citation resolver proves provenance and referential integrity;
it does not independently prove semantic entailment.
