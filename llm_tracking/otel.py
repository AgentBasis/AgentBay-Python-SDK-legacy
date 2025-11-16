"""Simple OpenTelemetry setup helpers for tracing (and optional metrics)."""

from __future__ import annotations

import os
from typing import Optional, Dict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPExporter
except Exception:  # pragma: no cover
    OTLPHTTPExporter = None  # type: ignore

# Default AgentBay backend endpoint
DEFAULT_AGENTBAY_ENDPOINT = "https://api.agentbay.com/v1/traces" # TODO: Make this configurable, this is a placeholder for now


def _get_endpoint(endpoint: Optional[str] = None) -> str:
    """
    Get endpoint with precedence:
    1. Explicit parameter
    2. OTEL_EXPORTER_OTLP_TRACES_ENDPOINT env var (standard OTel)
    3. OTEL_EXPORTER_OTLP_ENDPOINT env var (standard OTel, appends /v1/traces)
    4. AGENTBAY_ENDPOINT env var (custom)
    5. Default AgentBay endpoint
    """
    if endpoint:
        return endpoint
    
    # Check standard OpenTelemetry environment variables
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_endpoint:
        return traces_endpoint
    
    base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if base_endpoint:
        # Append /v1/traces if base URL provided
        return f"{base_endpoint.rstrip('/')}/v1/traces"
    
    # Check custom AgentBay env var
    agentbay_endpoint = os.getenv("AGENTBAY_ENDPOINT")
    if agentbay_endpoint:
        return agentbay_endpoint
    
    # Default to AgentBay backend
    return DEFAULT_AGENTBAY_ENDPOINT


def _get_api_key() -> Optional[str]:
    """
    Get API key from environment variable.
    
    Returns:
        API key if found, None otherwise
    """
    return os.getenv("AGENTBAY_API_KEY")


def _build_otlp_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Build headers for OTLP exporter.
    
    Args:
        api_key: API key to include in Authorization header
        
    Returns:
        Dictionary of headers
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def init_tracing(
    *, 
    exporter: str = "otlp-http", 
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None
) -> None:
    """Initialize a basic tracer provider with a chosen exporter.

    Args:
        exporter: "console" or "otlp-http" (default: "otlp-http")
        endpoint: OTLP HTTP endpoint (defaults to AgentBay backend or env var)
        api_key: API key for authentication (defaults to AGENTBAY_API_KEY env var)
    """
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    if exporter == "console":
        span_exporter = ConsoleSpanExporter()
    elif exporter == "otlp-http":
        if OTLPHTTPExporter is None:
            raise RuntimeError("OTLP HTTP exporter not available. Install opentelemetry-exporter-otlp.")
        
        # Get endpoint (from param, env var, or default)
        resolved_endpoint = _get_endpoint(endpoint)
        
        # Get API key (from param or env var)
        resolved_api_key = api_key or _get_api_key()
        
        # Build headers with API key
        headers = _build_otlp_headers(resolved_api_key)
        
        # Create exporter with endpoint and headers
        span_exporter = OTLPHTTPExporter(
            endpoint=resolved_endpoint,
            headers=headers if headers else None
        )
    else:
        raise ValueError(f"Unknown exporter: {exporter}")

    processor = BatchSpanProcessor(span_exporter)
    provider.add_span_processor(processor)

