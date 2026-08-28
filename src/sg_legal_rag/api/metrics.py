from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from sg_legal_rag.generation.production_contract import CitationContractIssueCode
from sg_legal_rag.generation.provider import TokenUsage
from sg_legal_rag.retrieval.artifacts import RetrievalArtifactIdentity

HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
RAG_DURATION_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
PROVIDER_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
RETRIEVAL_RESULT_BUCKETS = (0, 1, 2, 3, 5, 10)

FAILURE_CATEGORIES = frozenset(
    {
        "provider_failure",
        "malformed_generated_output",
        "citation_contract_failure",
        "evidence_integrity_failure",
        "retrieval_unavailable",
        "generation_unavailable",
        "request_validation_failure",
        "authentication_failure",
        "request_too_large",
        "context_budget_exceeded",
        "concurrency_limit",
        "provider_timeout",
        "internal_error",
    }
)
PROVIDER_FAILURE_CATEGORIES = frozenset(
    {"provider_failure", "provider_timeout", "malformed_generated_output"}
)
PROVIDER_NAMES = frozenset({"openai", "fake", "custom"})
MODEL_FAMILIES = frozenset({"gpt-5.6-luna", "fake", "custom"})
CITATION_ISSUE_CODES = frozenset(code.value for code in CitationContractIssueCode)


def bounded_provider_name(value: str) -> str:
    return value if value in PROVIDER_NAMES else "custom"


def bounded_model_family(value: str) -> str:
    return value if value in MODEL_FAMILIES else "custom"


class ApiMetrics:
    """Per-application Prometheus metrics with fixed, privacy-safe label domains."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "precedent_http_requests_total",
            "HTTP requests completed by method, route template, and status class.",
            ("method", "endpoint", "status_class"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "precedent_http_request_duration_seconds",
            "HTTP request duration by route template.",
            ("endpoint",),
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.requests_in_flight = Gauge(
            "precedent_requests_in_flight",
            "HTTP requests currently executing.",
            registry=self.registry,
        )
        self.rag_operations = Counter(
            "precedent_rag_operations_total",
            "Completed RAG operations by operation and outcome.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self.retrieval_duration = Histogram(
            "precedent_retrieval_duration_seconds",
            "Retrieval phase duration.",
            buckets=RAG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.generation_duration = Histogram(
            "precedent_generation_duration_seconds",
            "Generation phase duration.",
            buckets=RAG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.resolution_duration = Histogram(
            "precedent_resolution_duration_seconds",
            "Citation validation and source-resolution duration.",
            buckets=RAG_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.retrieval_results = Histogram(
            "precedent_retrieval_results",
            "Evidence passages returned by successful retrieval phases.",
            buckets=RETRIEVAL_RESULT_BUCKETS,
            registry=self.registry,
        )
        self.answer_outcomes = Counter(
            "precedent_answer_outcomes_total",
            "Successful answer operations by answered or insufficient-evidence outcome.",
            ("status",),
            registry=self.registry,
        )
        self.failures = Counter(
            "precedent_failures_total",
            "Operational failures by fixed application category.",
            ("category",),
            registry=self.registry,
        )
        self.citation_contract_violations = Counter(
            "precedent_citation_contract_violations_total",
            "Generated citation-contract issue instances by fixed issue code.",
            ("issue_code",),
            registry=self.registry,
        )
        provider_labels = ("provider", "model_family")
        self.provider_requests = Counter(
            "precedent_provider_requests_total",
            "Generation-provider request attempts.",
            provider_labels,
            registry=self.registry,
        )
        self.provider_failures = Counter(
            "precedent_provider_failures_total",
            "Generation-provider failures by fixed category.",
            (*provider_labels, "category"),
            registry=self.registry,
        )
        self.provider_duration = Histogram(
            "precedent_provider_duration_seconds",
            "Generation-provider request duration.",
            provider_labels,
            buckets=PROVIDER_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.llm_input_tokens = Counter(
            "precedent_llm_input_tokens_total",
            "Provider-reported input tokens.",
            provider_labels,
            registry=self.registry,
        )
        self.llm_output_tokens = Counter(
            "precedent_llm_output_tokens_total",
            "Provider-reported output tokens.",
            provider_labels,
            registry=self.registry,
        )
        self.llm_reasoning_tokens = Counter(
            "precedent_llm_reasoning_tokens_total",
            "Provider-reported reasoning tokens.",
            provider_labels,
            registry=self.registry,
        )
        self.llm_cached_input_tokens = Counter(
            "precedent_llm_cached_input_tokens_total",
            "Provider-reported cached input tokens.",
            provider_labels,
            registry=self.registry,
        )
        self.llm_estimated_cost_usd = Counter(
            "precedent_llm_estimated_cost_usd_total",
            "Estimated usage cost in USD; not reconciled provider billing.",
            provider_labels,
            registry=self.registry,
        )
        self.service_ready = Gauge(
            "precedent_service_ready",
            "Whether the service meets its retrieval-based readiness contract.",
            registry=self.registry,
        )
        self.retrieval_ready = Gauge(
            "precedent_retrieval_ready",
            "Whether prepared retrieval is available.",
            registry=self.registry,
        )
        self.generation_configured = Gauge(
            "precedent_generation_configured",
            "Whether generation credentials and a provider are configured.",
            registry=self.registry,
        )
        self.retrieval_documents = Gauge(
            "precedent_retrieval_documents",
            "Documents in the loaded immutable retrieval artifact.",
            registry=self.registry,
        )
        self.retrieval_artifact_load_seconds = Gauge(
            "precedent_retrieval_artifact_load_seconds",
            "Verified retrieval artifact load duration.",
            registry=self.registry,
        )
        self.retrieval_artifact_info = Gauge(
            "precedent_retrieval_artifact_info",
            "Immutable retrieval artifact identity without its full digest.",
            ("artifact_version",),
            registry=self.registry,
        )
        self.set_readiness(retrieval=False, generation=False)
        self.set_artifact(None)

    def begin_http_request(self) -> None:
        self.requests_in_flight.inc()

    def finish_http_request(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self.http_requests.labels(
            method=method,
            endpoint=endpoint,
            status_class=f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other",
        ).inc()
        self.http_duration.labels(endpoint=endpoint).observe(duration_seconds)
        self.requests_in_flight.dec()

    def record_rag_operation(self, *, operation: str, outcome: str) -> None:
        if operation not in {"retrieve", "answer"}:
            raise ValueError("RAG metric operation must be retrieve or answer")
        if outcome not in {"success", "failure"}:
            raise ValueError("RAG metric outcome must be success or failure")
        self.rag_operations.labels(operation=operation, outcome=outcome).inc()

    def observe_retrieval(self, duration_seconds: float, result_count: int | None = None) -> None:
        self.retrieval_duration.observe(duration_seconds)
        if result_count is not None:
            self.retrieval_results.observe(result_count)

    def observe_generation(self, duration_seconds: float) -> None:
        self.generation_duration.observe(duration_seconds)

    def observe_resolution(self, duration_seconds: float) -> None:
        self.resolution_duration.observe(duration_seconds)

    def record_answer_outcome(self, status: str) -> None:
        if status not in {"answered", "insufficient_evidence"}:
            raise ValueError("answer metric status is outside the production contract")
        self.answer_outcomes.labels(status=status).inc()

    def record_failure(self, category: str) -> None:
        bounded = category if category in FAILURE_CATEGORIES else "internal_error"
        self.failures.labels(category=bounded).inc()

    def record_citation_violations(self, issue_codes: Iterable[str]) -> None:
        for issue_code in issue_codes:
            if issue_code not in CITATION_ISSUE_CODES:
                raise ValueError("citation metric issue code is outside the production contract")
            self.citation_contract_violations.labels(issue_code=issue_code).inc()

    def begin_provider_request(self, *, provider: str, model_family: str) -> None:
        labels = (bounded_provider_name(provider), bounded_model_family(model_family))
        self.provider_requests.labels(provider=labels[0], model_family=labels[1]).inc()

    def finish_provider_request(
        self,
        *,
        provider: str,
        model_family: str,
        duration_seconds: float,
        failure_category: str | None = None,
    ) -> None:
        labels = {
            "provider": bounded_provider_name(provider),
            "model_family": bounded_model_family(model_family),
        }
        self.provider_duration.labels(**labels).observe(duration_seconds)
        if failure_category is not None:
            bounded = (
                failure_category
                if failure_category in PROVIDER_FAILURE_CATEGORIES
                else "provider_failure"
            )
            self.provider_failures.labels(**labels, category=bounded).inc()

    def record_usage(
        self,
        *,
        provider: str,
        model_family: str,
        usage: TokenUsage,
        estimated_cost_usd: float | None,
    ) -> None:
        labels = {
            "provider": bounded_provider_name(provider),
            "model_family": bounded_model_family(model_family),
        }
        self.llm_input_tokens.labels(**labels).inc(usage.input_tokens)
        self.llm_output_tokens.labels(**labels).inc(usage.output_tokens)
        self.llm_reasoning_tokens.labels(**labels).inc(usage.reasoning_tokens)
        self.llm_cached_input_tokens.labels(**labels).inc(usage.cached_input_tokens)
        if estimated_cost_usd is not None:
            self.llm_estimated_cost_usd.labels(**labels).inc(estimated_cost_usd)

    def set_readiness(self, *, retrieval: bool, generation: bool) -> None:
        self.service_ready.set(int(retrieval))
        self.retrieval_ready.set(int(retrieval))
        self.generation_configured.set(int(generation))

    def set_artifact(self, identity: RetrievalArtifactIdentity | None) -> None:
        self.retrieval_documents.set(identity.document_count if identity is not None else 0)
        self.retrieval_artifact_load_seconds.set(
            identity.load_ms / 1000 if identity is not None else 0
        )
        if identity is not None:
            self.retrieval_artifact_info.labels(artifact_version=identity.artifact_version).set(1)

    def render(self) -> bytes:
        return generate_latest(self.registry)
