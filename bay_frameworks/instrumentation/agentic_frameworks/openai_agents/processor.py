from typing import Any
from opentelemetry.trace import StatusCode
from MYSDK.bay_frameworks.logging import logger


class OpenAIAgentsProcessor:
	"""Processor for OpenAI Agents SDK traces."""

	def __init__(self, exporter=None):
		self.exporter = exporter

	def on_trace_start(self, sdk_trace: Any) -> None:
		logger.debug(f"[bay_frameworks.instrumentation.openai_agents] Trace started: {sdk_trace}")
		self.exporter.export_trace(sdk_trace)

	def on_trace_end(self, sdk_trace: Any) -> None:
		sdk_trace.status = StatusCode.OK.name
		logger.debug(f"[bay_frameworks.instrumentation.openai_agents] Trace ended: {sdk_trace}")
		self.exporter.export_trace(sdk_trace)

	def on_span_start(self, span: Any) -> None:
		logger.debug(f"[bay_frameworks.instrumentation.openai_agents] Span started: {span}")
		self.exporter.export_span(span)

	def on_span_end(self, span: Any) -> None:
		span.status = StatusCode.OK.name
		logger.debug(f"[bay_frameworks.instrumentation.openai_agents] Span ended: {span}")
		self.exporter.export_span(span)

	def shutdown(self) -> None:
		pass

	def force_flush(self) -> None:
		pass


