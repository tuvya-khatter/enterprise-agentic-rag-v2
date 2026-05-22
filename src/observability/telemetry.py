"""OpenTelemetry + Prometheus + structlog bootstrap."""
import logging
import sys

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram

from config.settings import get_settings

settings = get_settings()


# Prometheus metrics
QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    labelnames=["tenant_id", "status"],
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency in seconds",
    labelnames=["tenant_id"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
TOKEN_COUNTER = Counter(
    "rag_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["tenant_id", "direction"],  # direction: in|out
)
COST_COUNTER = Counter(
    "rag_cost_usd_total",
    "Estimated total LLM cost in USD",
    labelnames=["tenant_id"],
)


def setup_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
    )


def setup_tracing(app) -> None:
    """Wire up OTel tracing + FastAPI instrumentation."""
    resource = Resource.create({"service.name": "enterprise-agentic-rag"})
    provider = TracerProvider(resource=resource)
    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
