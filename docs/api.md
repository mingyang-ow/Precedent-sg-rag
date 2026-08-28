# FastAPI service

## Service boundary

The service exposes the production citation contract through a deliberately small HTTP surface:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness only |
| `GET` | `/ready` | Local retrieval and generation-configuration readiness |
| `GET` | `/metrics` | Prometheus metrics; separate metrics credential and internal placement |
| `GET` | `/version` | API, prompt, schema, and citation-contract identity |
| `POST` | `/retrieve` | Authenticated passage-BM25 evidence retrieval |
| `POST` | `/answer` | Authenticated retrieval, generation, validation, and source resolution |

Interactive OpenAPI documentation is available at `/docs` by default and can be disabled. The
service is not legal advice.

## Local startup

```bash
uv sync --extra dev --extra generation --extra api
uv run sg-legal-build-retrieval-artifacts
export PRECEDENT_API_KEY='replace-with-a-distinct-service-secret'
export PRECEDENT_METRICS_KEY='replace-with-a-distinct-metrics-secret'
uv run uvicorn sg_legal_rag.api.app:app --host 127.0.0.1 --port 8000
```

Then inspect:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/docs
```

Scrape metrics with `X-Precedent-Metrics-Key`; call `/retrieve` and `/answer` with
`X-Precedent-API-Key`. Health, readiness, version, docs, and OpenAPI remain public under the
default demo policy.

Service startup, `/health`, and `/ready` never call a model provider. `/answer` is unavailable when
generation is not configured.

## Configuration

Configuration comes directly from environment variables; a `.env` file is not required.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Enables `/answer`; never returned or logged |
| `PRECEDENT_API_KEY` | unset/fail closed | Credential for `/retrieve` and `/answer` |
| `PRECEDENT_METRICS_KEY` | unset/fail closed | Separate credential for `/metrics` |
| `MODEL_ID` | `gpt-5.6-luna` | Future production generation model |
| `TOP_K_DEFAULT` | `5` | Retrieval depth when omitted |
| `MAX_TOP_K` | `10` | Public retrieval-depth ceiling |
| `MAX_OUTPUT_TOKENS` | `600` | Provider output ceiling |
| `MAX_FACTS_CHARS` | `4000` | Facts character ceiling; cannot exceed the schema cap |
| `MAX_PRINCIPLE_CHARS` | `2000` | Principle character ceiling; cannot exceed the schema cap |
| `MAX_INPUT_TOKENS` | `16000` | Estimated prompt/query/evidence/schema ceiling |
| `MAX_CONCURRENT_GENERATIONS` | `2` | Non-queuing, process-local generation slots |
| `PROVIDER_TIMEOUT_SECONDS` | `30` | Provider timeout; automatic retries remain disabled |
| `REASONING_EFFORT` | `none` | Provider reasoning setting |
| `VERBOSITY` | `low` | Provider output verbosity |
| `ENABLE_DOCS` | `true` | Enables `/docs` and `/openapi.json` for demos |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost,testserver` | Trusted Host allow-list |
| `PRECEDENT_RETRIEVAL_ARTIFACTS` | `data/processed/retrieval-artifacts` | Prepared immutable retrieval bundle |

Readiness uses a partial policy. If local retrieval assets and configuration are valid but
`OPENAI_API_KEY` is absent, `/ready` returns HTTP 200 with `status: partial`: `/retrieve` is ready
and `/answer` returns 503. Missing retrieval assets produce HTTP 503 with `status: not_ready`.

The service, metrics, and OpenAI credentials must all be distinct; application credentials must
contain at least 16 characters. Protected routes return 503 if their required runtime credential
is not configured and 401 when the supplied credential is missing or invalid. Comparisons are
constant-time and credential values never enter errors or OpenAPI.

## Retrieval and answer flow

`POST /retrieve` accepts bounded facts, an optional principle, and an optional `top_k`. The service
canonicalizes the query and uses the verified passage corpus and restored BM25 state loaded during
application construction. It returns exact application-owned `EvidenceItem` text and provenance
and does not expose retriever tuning knobs. Normal requests never read the source CSV, reconstruct
the historical corpus, tokenize stored passages, or rebuild BM25.

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

Before a provider call, the service estimates tokens for the versioned system prompt,
deterministic query/evidence JSON, structured-output schema, and protocol overhead. It rejects an
over-budget context without truncation. A non-blocking process-local semaphore prevents provider
requests from accumulating in an unbounded executor queue; saturation returns 429. Provider calls
have an explicit timeout, map timeouts to 504, and use no automatic retries.

The artifact builder replaces the historical full-paragraph digest with a SHA-256 digest of the
exact bounded passage displayed by the production application. Historical frozen evidence and
signatures are not changed.

## Errors

Errors use a typed body containing a stable code, safe message, request ID, and optional issue
codes. Stack traces, raw provider errors, query text, evidence text, and credentials are not
returned.

| HTTP | Category | Examples |
| ---: | --- | --- |
| 401 | Authentication | Missing or invalid configured service/metrics credential |
| 400 | Request policy | `top_k` outside configured bounds |
| 413 | Resource policy | Character or estimated input-token limit exceeded |
| 429 | Capacity policy | Generation concurrency saturated |
| 422 | Request schema | Blank facts, wrong types, unknown fields |
| 503 | Dependency unavailable | Missing retrieval assets or generation configuration |
| 502 | Provider/generated output | Provider failure, malformed output, hallucinated or mismatched citation reference |
| 500 | Integrity/internal | Changed passage digest or impossible retrieved-evidence invariant |
| 504 | Provider availability | Explicit provider timeout |

Unknown evidence IDs, evidence/case mismatches, hidden evidence references, answers without claims,
and abstentions with citations are treated as invalid generated output (`502`). A passage-digest
mismatch means application-owned evidence changed and is therefore an internal integrity failure
(`500`). Nothing is silently repaired.

## Request identity and logging

The service accepts a safe `X-Request-ID` containing up to 64 ASCII letters, digits, dots,
underscores, or hyphens. Otherwise it generates a UUID. Every HTTP response returns the ID in the
same header, and structured error bodies include it.

JSON application logs contain request ID, route-template endpoint, method, HTTP status, total latency, retrieval
count, answer status, provider status, and retrieval/generation/resolution timings where relevant.
They deliberately omit full query and evidence text, API keys, and credential values. A formal
metrics backend can scrape the privacy-safe `/metrics` endpoint described in
[`observability.md`](observability.md). Request IDs remain log-only and never become metric labels.

CORS is disabled. Trusted hosts are allow-listed. Responses add `X-Content-Type-Options: nosniff`
and `Cache-Control: no-store`; HSTS is left to the TLS termination boundary. The complete threat
model and residual risks are in [`security.md`](security.md).

## Bounded technical debt

Evidence IDs remain local to one package (`E1`, `E2`, ...). Responses therefore retain the
top-level `package_id`, while citations include `evidence_id` and `passage_digest`. A persistent
production evidence store remains deferred. The citation resolver proves provenance and
referential integrity; it does not independently prove semantic entailment. Authentication is
shared service-level access, and concurrency is process-local rather than a distributed rate
limit. See
[`deployment.md`](deployment.md) for the artifact and container contract.
