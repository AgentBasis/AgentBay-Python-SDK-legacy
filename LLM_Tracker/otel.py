"""Simple OpenTelemetry setup helpers for tracing (and optional metrics)."""

from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPExporter
except Exception:  # pragma: no cover
    OTLPHTTPExporter = None  # type: ignore


def init_tracing(*, exporter: str = "console", endpoint: Optional[str] = None) -> None:
    """Initialize a basic tracer provider with a chosen exporter.

    Args:
        exporter: "console" (default) or "otlp-http"
        endpoint: OTLP HTTP endpoint (e.g., https://otlp.example.com/v1/traces)
    """
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    if exporter == "console":
        span_exporter = ConsoleSpanExporter()
    elif exporter == "otlp-http":
        if OTLPHTTPExporter is None:
            raise RuntimeError("OTLP HTTP exporter not available. Install opentelemetry-exporter-otlp.")
        if not endpoint:
            raise ValueError("endpoint is required for otlp-http exporter")
        span_exporter = OTLPHTTPExporter(endpoint=endpoint)
    else:
        raise ValueError(f"Unknown exporter: {exporter}")

    processor = BatchSpanProcessor(span_exporter)
    provider.add_span_processor(processor)

