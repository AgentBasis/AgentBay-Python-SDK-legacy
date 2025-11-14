import json
from typing import Any, Dict, Optional

from opentelemetry import trace, context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode, NonRecordingSpan
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import Span

from agentbay.bay_frameworks.logging import logger
from agentbay.bay_frameworks.semconv import CoreAttributes

from agentbay.bay_frameworks.instrumentation.common.attributes import (
	get_base_trace_attributes,
	get_base_span_attributes,
)

from agentbay.bay_frameworks.instrumentation.agentic_frameworks.openai_agents import LIBRARY_NAME, LIBRARY_VERSION
from agentbay.bay_frameworks.instrumentation.agentic_frameworks.openai_agents.attributes.common import (
	get_span_attributes,
)


def log_otel_trace_id(span_type):
	current_span = trace.get_current_span()
	if hasattr(current_span, "get_span_context"):
		ctx = current_span.get_span_context()
		if hasattr(ctx, "trace_id") and ctx.trace_id:
			otel_trace_id = f"{ctx.trace_id:032x}" if isinstance(ctx.trace_id, int) else str(ctx.trace_id)
			logger.debug(f"[SPAN] Export | Type: {span_type} | TRACE ID: {otel_trace_id}")
			return otel_trace_id
	logger.debug(f"[SPAN] Export | Type: {span_type} | NO TRACE ID AVAILABLE")
	return None


def get_span_kind(span: Any) -> SpanKind:
	span_data = span.span_data
	span_type = span_data.__class__.__name__
	if span_type == "AgentSpanData":
		return SpanKind.CONSUMER
	elif span_type in ["FunctionSpanData", "GenerationSpanData", "ResponseSpanData"]:
		return SpanKind.CLIENT
	else:
		return SpanKind.INTERNAL


def get_span_name(span: Any) -> str:
	span_data = span.span_data
	span_type = span_data.__class__.__name__
	if hasattr(span_data, "name") and span_data.name:
		return span_data.name
	else:
		return span_type.replace("SpanData", "").lower()


def _get_span_lookup_key(trace_id: str, span_id: str) -> str:
	return f"span:{trace_id}:{span_id}"


class OpenAIAgentsExporter:
	def __init__(self, tracer_provider=None):
		self.tracer_provider = tracer_provider
		self._active_spans = {}
		self._span_map = {}

	def export_trace(self, trace_obj: Any) -> None:
		tracer = get_tracer(LIBRARY_NAME, LIBRARY_VERSION, self.tracer_provider)
		trace_id = getattr(trace_obj, "trace_id", "unknown")
		if not hasattr(trace_obj, "trace_id"):
			logger.debug("Cannot export trace: missing trace_id")
			return
		is_end_event = hasattr(trace_obj, "status") and trace_obj.status == StatusCode.OK.name
		trace_lookup_key = _get_span_lookup_key(trace_id, trace_id)
		attributes = get_base_trace_attributes(trace_obj)
		if is_end_event and trace_lookup_key in self._span_map:
			existing_span = self._span_map[trace_lookup_key]
			span_is_ended = False
			if isinstance(existing_span, Span) and hasattr(existing_span, "_end_time"):
				span_is_ended = existing_span._end_time is not None
			if not span_is_ended:
				for key, value in attributes.items():
					existing_span.set_attribute(key, value)
				if hasattr(trace_obj, "error") and trace_obj.error:
					self._handle_span_error(trace_obj, existing_span)
				else:
					existing_span.set_status(Status(StatusCode.OK))
				existing_span.end()
				self._active_spans.pop(trace_id, None)
				self._span_map.pop(trace_lookup_key, None)
				return
		span = tracer.start_span(name=trace_obj.name, kind=SpanKind.INTERNAL, attributes=attributes)
		if hasattr(trace_obj, "group_id") and trace_obj.group_id:
			span.set_attribute(CoreAttributes.GROUP_ID, trace_obj.group_id)
		if hasattr(trace_obj, "metadata") and trace_obj.metadata:
			for key, value in trace_obj.metadata.items():
				if isinstance(value, (str, int, float, bool)):
					span.set_attribute(f"trace.metadata.{key}", value)
		if hasattr(trace_obj, "error") and trace_obj.error:
			self._handle_span_error(trace_obj, span)
		if not is_end_event:
			self._span_map[trace_lookup_key] = span
			self._active_spans[trace_id] = {"span": span, "span_type": "TraceSpan", "trace_id": trace_id, "parent_id": None}
		else:
			span.end()

	def _get_parent_context(self, trace_id: str, span_id: str, parent_id: Optional[str] = None) -> Any:
		parent_span_ctx = None
		if parent_id:
			parent_lookup_key = f"span:{trace_id}:{parent_id}"
			if parent_lookup_key in self._span_map:
				parent_span = self._span_map[parent_lookup_key]
				if hasattr(parent_span, "get_span_context"):
					parent_span_ctx = parent_span.get_span_context()
		if not parent_span_ctx and parent_id is None:
			trace_lookup_key = _get_span_lookup_key(trace_id, trace_id)
			if trace_lookup_key in self._span_map:
				trace_span = self._span_map[trace_lookup_key]
				if hasattr(trace_span, "get_span_context"):
					parent_span_ctx = trace_span.get_span_context()
		if not parent_span_ctx:
			ctx = context_api.get_current()
			parent_span_ctx = trace_api.get_current_span(ctx).get_span_context()
		return parent_span_ctx

	def _create_span_with_parent(self, name: str, kind: SpanKind, attributes: Dict[str, Any], parent_ctx: Any, end_immediately: bool = False) -> Any:
		tracer = get_tracer(LIBRARY_NAME, LIBRARY_VERSION, self.tracer_provider)
		with trace_api.use_span(NonRecordingSpan(parent_ctx), end_on_exit=False):
			span = tracer.start_span(name=name, kind=kind, attributes=attributes)
		if end_immediately:
			span.end()
		return span

	def export_span(self, span: Any) -> None:
		if not hasattr(span, "span_data"):
			return
		span_data = span.span_data
		span_type = span_data.__class__.__name__
		span_id = getattr(span, "span_id", "unknown")
		trace_id = getattr(span, "trace_id", "unknown")
		parent_id = getattr(span, "parent_id", None)
		is_end_event = hasattr(span, "status") and span.status == StatusCode.OK.name
		span_lookup_key = _get_span_lookup_key(trace_id, span_id)
		attributes = get_base_span_attributes(span)
		span_attributes = get_span_attributes(span_data)
		attributes.update(span_attributes)
		if is_end_event:
			attributes.update(span_attributes)
		log_otel_trace_id(span_type)
		if not is_end_event:
			span_name = get_span_name(span)
			span_kind = get_span_kind(span)
			parent_span_ctx = self._get_parent_context(trace_id, span_id, parent_id)
			otel_span = self._create_span_with_parent(name=span_name, kind=span_kind, attributes=attributes, parent_ctx=parent_span_ctx)
			if not isinstance(otel_span, NonRecordingSpan):
				self._span_map[span_lookup_key] = otel_span
				self._active_spans[span_id] = {"span": otel_span, "span_type": span_type, "trace_id": trace_id, "parent_id": parent_id}
			self._handle_span_error(span, otel_span)
			return
		if span_lookup_key in self._span_map:
			existing_span = self._span_map[span_lookup_key]
			span_is_ended = False
			if isinstance(existing_span, Span) and hasattr(existing_span, "_end_time"):
				span_is_ended = existing_span._end_time is not None
			if not span_is_ended:
				for key, value in attributes.items():
					existing_span.set_attribute(key, value)
				existing_span.set_status(Status(StatusCode.OK if getattr(span, "status", "OK") == "OK" else StatusCode.ERROR))
				self._handle_span_error(span, existing_span)
				existing_span.end()
			else:
				logger.warning(
					f"[Exporter] SDK span_id: {span_id} (END event) - Attempting to end an ALREADY ENDED span: {span_lookup_key}. Creating a new one instead."
				)
				self.create_span(span, span_type, attributes, is_already_ended=True)
		else:
			logger.warning(
				f"[Exporter] SDK span_id: {span_id} (END event) - No active span found for end event: {span_lookup_key}. Creating a new one."
			)
			self.create_span(span, span_type, attributes, is_already_ended=True)
		self._active_spans.pop(span_id, None)
		self._span_map.pop(span_lookup_key, None)

	def create_span(self, span: Any, span_type: str, attributes: Dict[str, Any], is_already_ended: bool = False) -> None:
		parent_ctx = None
		if hasattr(span, "parent_id") and span.parent_id:
			parent_ctx = self._get_parent_context(getattr(span, "trace_id", "unknown"), getattr(span, "id", "unknown"), span.parent_id)
		name = get_span_name(span)
		kind = get_span_kind(span)
		self._create_span_with_parent(name=name, kind=kind, attributes=attributes, parent_ctx=parent_ctx, end_immediately=True)

	def _handle_span_error(self, span: Any, otel_span: Any) -> None:
		if hasattr(span, "error") and span.error:
			status = Status(StatusCode.ERROR)
			otel_span.set_status(status)
			error_message = "Unknown error"
			error_data = {}
			error_type = "AgentError"
			if isinstance(span.error, dict):
				error_message = span.error.get("message", span.error.get("error", "Unknown error"))
				error_data = span.error.get("data", {})
				if "type" in span.error:
					error_type = span.error["type"]
				elif "code" in span.error:
					error_type = span.error["code"]
			elif isinstance(span.error, str):
				error_message = span.error
			elif hasattr(span.error, "message"):
				error_message = span.error.message
				error_type = type(span.error).__name__
			elif hasattr(span.error, "__str__"):
				error_message = str(span.error)
			try:
				exception = Exception(error_message)
				error_data_json = json.dumps(error_data) if error_data else "{}"
				otel_span.record_exception(exception=exception, attributes={"error.data": error_data_json})
			except Exception as e:
				logger.warning(f"Error serializing error data: {e}")
				otel_span.record_exception(Exception(error_message))
			otel_span.set_attribute(CoreAttributes.ERROR_TYPE, error_type)
			otel_span.set_attribute(CoreAttributes.ERROR_MESSAGE, error_message)

	def cleanup(self):
		self._active_spans.clear()
		self._span_map.clear()


