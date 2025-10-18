"""Common span management utilities for bay_frameworks instrumentation."""

import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, Tuple
from functools import wraps

from opentelemetry.trace import Tracer, Span, SpanKind, Status, StatusCode, get_current_span
from opentelemetry.sdk.resources import SERVICE_NAME, TELEMETRY_SDK_NAME, DEPLOYMENT_ENVIRONMENT
from opentelemetry import context as context_api

from MYSDK.bay_frameworks.logging import logger
from MYSDK.bay_frameworks.semconv import CoreAttributes


class SpanAttributeManager:
	"""Manages common span attributes across instrumentations."""

	def __init__(self, service_name: str = "bay_frameworks", deployment_environment: str = "production"):
		self.service_name = service_name
		self.deployment_environment = deployment_environment

	def set_common_attributes(self, span: Span):
		span.set_attribute(TELEMETRY_SDK_NAME, "bay_frameworks")
		span.set_attribute(SERVICE_NAME, self.service_name)
		span.set_attribute(DEPLOYMENT_ENVIRONMENT, self.deployment_environment)

	def set_config_tags(self, span: Span):
		# No global client in bay_frameworks; tags can be added by callers if desired
		pass


@contextmanager
def create_span(
	tracer: Tracer,
	name: str,
	kind: SpanKind = SpanKind.CLIENT,
	attributes: Optional[Dict[str, Any]] = None,
	set_common_attributes: bool = True,
	attribute_manager: Optional[SpanAttributeManager] = None,
):
	with tracer.start_as_current_span(name, kind=kind, attributes=attributes or {}) as span:
		try:
			if set_common_attributes and attribute_manager:
				attribute_manager.set_common_attributes(span)
			yield span
			span.set_status(Status(StatusCode.OK))
		except Exception as e:
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			logger.error(f"Error in span {name}: {e}")
			raise


def timed_span(tracer: Tracer, name: str, record_duration: Optional[Callable[[float], None]] = None, **span_kwargs):
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			start_time = time.time()
			with create_span(tracer, name, **span_kwargs):
				result = func(*args, **kwargs)
				if record_duration:
					duration = time.time() - start_time
					record_duration(duration)
				return result

		return wrapper

	return decorator


class StreamingSpanManager:
	def __init__(self, tracer: Tracer):
		self.tracer = tracer
		self._active_spans: Dict[Any, Span] = {}

	def start_streaming_span(self, stream_id: Any, name: str, **span_kwargs) -> Span:
		span = self.tracer.start_span(name, **span_kwargs)
		self._active_spans[stream_id] = span
		return span

	def get_streaming_span(self, stream_id: Any) -> Optional[Span]:
		return self._active_spans.get(stream_id)

	def end_streaming_span(self, stream_id: Any, status: Optional[Status] = None):
		span = self._active_spans.pop(stream_id, None)
		if span:
			if status:
				span.set_status(status)
			else:
				span.set_status(Status(StatusCode.OK))
			span.end()


def extract_parent_context(parent_span: Optional[Span] = None) -> Any:
	if parent_span:
		from opentelemetry.trace import set_span_in_context

		return set_span_in_context(parent_span)
	return context_api.get_current()


def safe_set_attribute(span: Span, key: str, value: Any, max_length: int = 1000):
	if value is None:
		return
	if isinstance(value, str) and len(value) > max_length:
		value = value[: max_length - 3] + "..."
	try:
		span.set_attribute(key, value)
	except Exception as e:
		logger.debug(f"Failed to set span attribute {key}: {e}")


def get_span_context_info(span: Optional[Span] = None) -> Tuple[str, str]:
	if not span:
		span = get_current_span()
	span_context = span.get_span_context()
	trace_id = format(span_context.trace_id, "032x") if span_context.trace_id else "unknown"
	span_id = format(span_context.span_id, "016x") if span_context.span_id else "unknown"
	return trace_id, span_id


