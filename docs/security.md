# Security and abuse resistance

Precedent treats the user query, retrieved legal text, and provider output as untrusted. The
application—not the model—enforces authentication, resource limits, the output schema, evidence
visibility, case/evidence consistency, passage integrity, and response construction. These are
practical portfolio controls, not a claim that the service is secure or suitable for legal-system
deployment.

## Assets and actors

Assets include the OpenAI credential, service and metrics credentials, provider budget, frozen
retrieval artifacts, source/evidence integrity, production prompt and policies, generated
responses, and operational logs and metrics.

The practical threat actors are a malicious API caller, a curious or abusive user, malicious text
embedded in a retrieved document, malformed or compromised provider output, and an operator who
accidentally misconfigures the service. Nation-state and bespoke hardware threats are outside this
model.

## Trust boundaries

```text
untrusted client
    |  service credential + typed bounded request
    v
FastAPI policy boundary
    |  canonical query
    v
digest-verified, read-only retrieval bundle
    |  bounded evidence (still untrusted text)
    v
versioned production prompt + structured provider request
    |  untrusted ProductionAnswer proposal
    v
strict schema + evidence-ID visibility allow-list + case/digest checks
    |  application-resolved source text
    v
typed response + bounded telemetry
```

The specific crossings are Internet/client to FastAPI, request to retrieval, retrieved evidence to
model context, provider to application, application to the frozen evidence resolver, and container
to its externally mounted artifact bundle. The LLM is not a trust boundary.

## Threat, control, and residual risk

| Threat | Enforced control | Residual risk |
| --- | --- | --- |
| User prompt injection | Fixed `rag-production-v2` policy; query is nested under `untrusted_data`; model receives no tools | Prompting cannot eliminate semantic manipulation |
| Retrieved-content injection | Evidence remains quoted JSON data; prompt denies document authority, tools, secrets, and schema changes | A model can still be influenced by adversarial legal prose |
| Hallucinated or invisible evidence ID | Package-local visible-ID allow-list | A visible passage may still be semantically weak |
| Case/evidence mismatch or invalid status | Strict `production-citation-v1` validation | Referential integrity is not entailment |
| Model-controlled source text | Extra fields rejected; backend resolves the exact stored passage | Client presentation must preserve this distinction |
| Artifact tampering | Fixed file names, manifest/file/payload/passage SHA-256 checks, `allow_pickle=False`, symlink rejection | A trusted publisher can replace a whole valid bundle; local TOCTOU is deployment risk |
| Secret leakage | Runtime-only `SecretStr` settings; fixed safe errors/logs/metric labels; regression sentinels | Host, proxy, and provider handling are outside the process |
| Unauthorized business use | Dedicated application key, constant-time comparison, fail-closed configuration | Service key is shared service identity, not end-user identity |
| Metrics disclosure | Separate metrics key plus intended internal network placement | Metrics reveal bounded operational state to authorized scrapers |
| Oversized or expensive requests | Character, top-k, context-token, output-token, concurrency, and timeout bounds | No per-caller quota or monetary budget ledger |
| Concurrent cost abuse | Non-blocking process-local semaphore; saturation returns 429 | Limits are not distributed across replicas |
| Provider hangs or duplicate cost | Explicit timeout and zero automatic retries | Provider-side work may outlive a client timeout |
| Malformed/compromised provider output | Strict Pydantic schema and frozen evidence resolver | Semantically plausible unsupported claims need an independent evaluator |
| Sensitive telemetry leakage | Fixed-cardinality labels and allow-lists; no query/evidence/model text | Infrastructure may add logs beyond this application |
| Host-header abuse | Configurable `TrustedHostMiddleware` allow-list | Reverse proxy configuration must preserve the intended host |
| Browser cross-origin abuse | CORS disabled | A future browser UI must configure explicit origins at its deployment boundary |

## Access policy

The split is intentional and keeps the portfolio demo useful without leaving costly routes open:

| Surface | Default policy | Reason |
| --- | --- | --- |
| `/health`, `/ready` | Public | Orchestrator probes; responses contain bounded state only |
| `/version` | Public | Contract and artifact identity for reproducibility; no path or secret |
| `/docs`, `/openapi.json` | Public when `ENABLE_DOCS=true`; removable | Useful demo surface; disable in a hardened deployment |
| `/retrieve`, `/answer` | `X-Precedent-API-Key` | Business/cost surface |
| `/metrics` | Separate `X-Precedent-Metrics-Key` and internal network | Operational information has a distinct consumer and lifecycle |

`PRECEDENT_API_KEY`, `PRECEDENT_METRICS_KEY`, and `OPENAI_API_KEY` must be distinct. Application
credentials must contain at least 16 characters. Missing configured credentials make their routes
fail closed with 503; missing or incorrect supplied credentials receive a fixed 401. Values never
enter OpenAPI examples, exception details, logs, or metrics. This is deliberately simple service
authentication, not users, OAuth, JWTs, roles, or authorization accounting.

## Abuse and cost boundaries

Defaults are 4,000 facts characters, 2,000 principle characters, top-k at most 10, 16,000 estimated
provider-input tokens, 600 output tokens, two simultaneous generations, a 30-second provider
timeout, and zero automatic provider retries. The input estimate counts the system prompt,
deterministic query/evidence envelope, output schema, and fixed protocol overhead with the local
tokenizer before provider execution. Oversized text is never silently truncated.

The deterministic maximum configured provider budget per accepted generation is therefore 16,000
input plus 600 output tokens (16,600 total), subject to provider tokenizer/accounting differences.
Retrieval and generation have additional top-k and character bounds. Prometheus records actual
provider-reported tokens and estimated cost, but the service intentionally has no dollar billing
ledger or per-caller quota.

Generation admission is non-queuing: if the process-local semaphore is full, `/answer` returns 429
with `Retry-After`. This deployment uses one process. Multi-instance concurrency/rate enforcement,
DDoS resistance, and quotas belong at a gateway or deployment layer.

## Prompt and evidence isolation

`rag-production-v2` changes only the never-yet-executed production prompt. Historical `rag-v2`
experiments, cached outputs, and adjudications remain untouched. The provider input is
deterministic JSON with one `untrusted_data` object containing the query and ordered evidence.
Malicious braces, fake JSON, role claims, and “ignore instructions” strings stay escaped string
values. This improves separation but is defense-in-depth, not a model sandbox or injection
detector.

The application records only objective enforcement outcomes. It does not emit a speculative
“prompt injection detected” signal.

## Security regression coverage

The offline fixture corpus covers prompt overrides, secret requests, fabricated evidence IDs,
citation suppression, fake system/developer content, JSON-envelope escape attempts, and unrelated
case steering. Fake outputs cover unknown and invisible IDs, changed case IDs, model-supplied
source text, extra fields, answers without claims, and citations on abstention.

Tests are grouped conceptually as:

- **AUTH:** public/protected route split, key separation, fail-closed configuration, constant-time
  verifier behavior, and configurable docs.
- **INPUT:** absolute and configured character limits, schema rejection, host/request-ID handling.
- **PROMPT_INJECTION / EVIDENCE_INJECTION:** deterministic untrusted-data envelope and policy text.
- **OUTPUT_VALIDATION:** strict schema and status/reference/visibility/case/digest rules.
- **SECRET_LEAKAGE:** sentinel values absent from HTTP, logs, metrics, and OpenAPI.
- **RESOURCE_ABUSE:** token budget, non-queuing concurrency, timeout, and fixed telemetry.
- **ARTIFACT_INTEGRITY:** missing/corrupt/path-traversal/symlink rejection and read-only loading.

Run the focused injection/output corpus with `uv run pytest tests/security/`; the complete suite
contains the surrounding API, resource, secret, and artifact regressions. No test calls a model
provider.

## HTTP and deployment posture

CORS is disabled because no browser frontend exists. All application responses add
`X-Content-Type-Options: nosniff` and `Cache-Control: no-store`. Invalid request IDs are replaced,
not reflected. HSTS is deliberately absent because TLS termination is outside this application.
Allowed hosts are explicit and configurable with `ALLOWED_HOSTS`.

The image remains non-root (`10001:10001`) and read-only capable. Artifacts are a separate
read-only mount and publish as traversable `0755` directories with immutable readable `0444`
files. The image contains neither artifacts nor credentials. Docker CI checks public health,
protected endpoints, separate metrics access, image metadata/history for sentinel secrets,
non-root execution, read-only root state, and read-only artifact access.

CI also runs one maintained Python dependency scanner, `pip-audit`, against the locked environment.
The first audit found eight advisory records against transitive Starlette 0.48.0, including malformed
URL interpretation and denial-of-service classes. FastAPI was advanced from 0.119.1 to the tested
0.141.1 compatibility line and Starlette to 1.6.0 (above the highest listed 1.3.1 fix). The full API
suite passed and the repeat audit reported no known vulnerabilities. Findings are not suppressed;
future changes must assess them rather than apply unreviewed upgrades.

## Known limitations

- Prompt injection cannot be eliminated solely through prompting.
- Retrieved legal documents remain capable of semantic manipulation even when treated as data.
- Deterministic citation controls prove identity, visibility, and integrity—not legal correctness or
  semantic entailment.
- Service keys do not provide end-user identity, rotation automation, audit attribution, or scopes.
- Concurrency and metrics are process-local; distributed limiting and aggregation are deployment
  concerns.
- TLS, reverse-proxy policy, network isolation, DDoS controls, and secret management are external.
- There is no WAF, distributed rate limiter, per-caller quota, or billing cutoff.
- No claim of legal-system-grade security is made. The Phase 7.6 independent semantic judge stays
  out of band and does not become a security trust boundary.
