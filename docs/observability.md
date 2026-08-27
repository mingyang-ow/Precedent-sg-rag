# Operational observability

Precedent exposes privacy-safe, Prometheus-compatible runtime metrics at `GET /metrics`. Scraping
metrics does not run retrieval or contact the generation provider. The endpoint requires no
writable state and works in the existing non-root, read-only container.

`/metrics` is intended for an internal monitoring network in production and requires the separate
`PRECEDENT_METRICS_KEY` through `X-Precedent-Metrics-Key`. This is defense-in-depth alongside
reverse-proxy, service-mesh, or network-policy restrictions; it is deliberately distinct from the
business endpoint credential.

## Scraping

A minimal Prometheus job is:

```yaml
scrape_configs:
  - job_name: precedent
    metrics_path: /metrics
    static_configs:
      - targets: ["precedent:8000"]
```

Configure the scraper or an internal reverse proxy to add `X-Precedent-Metrics-Key` from a runtime
secret; do not put the credential directly in this checked-in example.

For a local check:

```bash
curl --silent --header "X-Precedent-Metrics-Key: $PRECEDENT_METRICS_KEY" \
  http://127.0.0.1:8000/metrics | grep '^precedent_'
```

The repository includes a datasource-agnostic Grafana dashboard at
[`observability/grafana-dashboard.json`](../observability/grafana-dashboard.json). Import it and
select the Prometheus datasource when prompted. No Prometheus or Grafana infrastructure is needed
by the application or CI.

## Metric catalog

| Metric | Type | Meaning |
| --- | --- | --- |
| `precedent_http_requests_total` | Counter | Requests by bounded method, route template, and status class |
| `precedent_http_request_duration_seconds` | Histogram | End-to-end HTTP latency by route template |
| `precedent_requests_in_flight` | Gauge | Requests currently executing, including a scrape while it is rendered |
| `precedent_rag_operations_total` | Counter | Successful or failed `/retrieve` and `/answer` operations |
| `precedent_retrieval_duration_seconds` | Histogram | Retrieval phase latency, including failed attempts |
| `precedent_generation_duration_seconds` | Histogram | Generation phase latency, including failed attempts |
| `precedent_resolution_duration_seconds` | Histogram | Citation validation and application-owned source resolution latency |
| `precedent_retrieval_results` | Histogram | Number of evidence passages returned by successful retrieval |
| `precedent_answer_outcomes_total` | Counter | Successful answers split into `answered` and `insufficient_evidence` |
| `precedent_failures_total` | Counter | Stable operational failure taxonomy |
| `precedent_citation_contract_violations_total` | Counter | Rejected generated references by contract issue code |
| `precedent_provider_requests_total` | Counter | Provider request attempts |
| `precedent_provider_failures_total` | Counter | Provider API or malformed-output failures |
| `precedent_provider_duration_seconds` | Histogram | Provider request latency |
| `precedent_llm_*_tokens_total` | Counter | Reliable provider-reported input, cached input, output, and reasoning usage |
| `precedent_llm_estimated_cost_usd_total` | Counter | Estimated usage cost, not reconciled billing |
| `precedent_service_ready` | Gauge | `1` when the retrieval-based readiness contract is met |
| `precedent_retrieval_ready` | Gauge | `1` when prepared retrieval is available |
| `precedent_generation_configured` | Gauge | `1` when a generation provider is configured |
| `precedent_retrieval_documents` | Gauge | Documents in the loaded retrieval bundle |
| `precedent_retrieval_artifact_load_seconds` | Gauge | Verified artifact load time |
| `precedent_retrieval_artifact_info` | Gauge | Artifact version identity; the full digest is deliberately omitted |

Prometheus histograms expose `_bucket`, `_count`, and `_sum` series. This supports p50, p95, and
p99 calculations without recording one series per request.

## Failure interpretation

`precedent_failures_total` has a fixed `category` set:

```text
provider_failure
malformed_generated_output
citation_contract_failure
evidence_integrity_failure
retrieval_unavailable
generation_unavailable
request_validation_failure
authentication_failure
request_too_large
context_budget_exceeded
concurrency_limit
provider_timeout
internal_error
```

Abstention is a successful application outcome, not a failure. It increments
`precedent_answer_outcomes_total{status="insufficient_evidence"}`.

A provider request can succeed while the application rejects its answer. In that case provider
request, duration, usage, and estimated-cost telemetry are retained, while
`precedent_failures_total{category="citation_contract_failure"}` and the relevant fixed citation
issue codes increment. An evidence digest mismatch is instead an application-owned integrity
failure.

Provider and model labels are normalized to a small allow-list. Unknown injected providers or
model families become `custom`; arbitrary returned model revisions never become labels. Token
counters increment only when the provider supplies usage. The cost counter uses the centralized,
frozen model-price snapshot in `generation/pricing.py`; it is an estimate and does not query or
reconcile a billing API.

## Operational questions

Example PromQL expressions answer useful questions without inventing production SLO targets:

```promql
# p95 /answer HTTP latency over five minutes
histogram_quantile(
  0.95,
  sum by (le) (rate(precedent_http_request_duration_seconds_bucket{endpoint="/answer"}[5m]))
)

# 5xx request fraction
sum(rate(precedent_http_requests_total{status_class="5xx"}[5m]))
/
sum(rate(precedent_http_requests_total[5m]))

# abstention fraction among successful answer outcomes
rate(precedent_answer_outcomes_total{status="insufficient_evidence"}[15m])
/
sum(rate(precedent_answer_outcomes_total[15m]))

# provider failure fraction
sum(rate(precedent_provider_failures_total[15m]))
/
sum(rate(precedent_provider_requests_total[15m]))

# citation-contract rejection rate and retrieval readiness
sum(rate(precedent_failures_total{category="citation_contract_failure"}[15m]))
precedent_retrieval_ready
```

## Structured logs and privacy

JSON logs retain bounded operational fields such as request ID, route template, HTTP status,
latency, retrieval count, answer status, provider status, fixed error code, and
retrieval/generation/resolution timings. Request IDs remain useful for log correlation but are
never metric labels.

Metrics and default logs deliberately exclude:

- API keys and credential values;
- query, principle, prompt, evidence, and model-response text;
- case names, evidence IDs, package IDs, and passage digests;
- request IDs as metric labels;
- exception messages or arbitrary provider revisions as metric labels.

The HTTP endpoint label comes from FastAPI's registered route template. Unknown paths become the
single `unmatched` value. HTTP methods, failure categories, provider identities, answer outcomes,
and citation issue codes are likewise bounded.

## Runtime boundaries

Metrics are process-local. Counters reset on restart and Prometheus should sum across replicas.
The current deployment intentionally uses one Uvicorn worker; a future multi-process deployment
would require the Prometheus client's multi-process mode or an external aggregation design. Phase
7 does not add distributed tracing, a collector, log aggregation, or alerting infrastructure.
Authentication is service-level only; network isolation and secret rotation remain deployment
responsibilities.
