from typing import Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from MYSDK.bay_frameworks.instrumentation.agentic_frameworks.openai_agents import LIBRARY_VERSION

from MYSDK.bay_frameworks.logging import logger
from MYSDK.bay_frameworks.instrumentation.agentic_frameworks.openai_agents.processor import OpenAIAgentsProcessor
from MYSDK.bay_frameworks.instrumentation.agentic_frameworks.openai_agents.exporter import OpenAIAgentsExporter


class OpenAIAgentsInstrumentor(BaseInstrumentor):
	"""An instrumentor for OpenAI Agents SDK that uses the built-in tracing API."""

	_processor = None
	_exporter = None
	_default_processor = None

	def __init__(self):
		super().__init__()
		self._tracer = None
		self._is_instrumented_instance_flag = False

	def instrumentation_dependencies(self) -> Collection[str]:
		return ["openai-agents >= 0.0.1"]

	def _instrument(self, **kwargs):
		if self._is_instrumented_instance_flag:
			logger.debug("OpenAI Agents SDK already instrumented. Skipping.")
			return

		tracer_provider = kwargs.get("tracer_provider")
		if self._tracer is None:
			logger.debug("OpenAI Agents SDK tracer is None, creating new tracer.")
			self._tracer = trace.get_tracer("bay_frameworks.instrumentation.openai_agents", LIBRARY_VERSION)

		try:
			self._exporter = OpenAIAgentsExporter(tracer_provider=tracer_provider)
			self._processor = OpenAIAgentsProcessor(exporter=self._exporter)

			from agents import set_trace_processors
			from agents.tracing.processors import default_processor

			if getattr(self, "_default_processor", None) is None:
				self._default_processor = default_processor()

			set_trace_processors([self._processor])
			self._is_instrumented_instance_flag = True

		except Exception as e:
			logger.warning(f"Failed to instrument OpenAI Agents SDK: {e}", exc_info=True)

	def _uninstrument(self, **kwargs):
		if not self._is_instrumented_instance_flag:
			logger.debug("OpenAI Agents SDK not currently instrumented. Skipping uninstrument.")
			return
		try:
			if hasattr(self, "_exporter") and self._exporter:
				if hasattr(self._exporter, "cleanup"):
					self._exporter.cleanup()
			from agents import set_trace_processors
			if hasattr(self, "_default_processor") and self._default_processor:
				set_trace_processors([self._default_processor])
				self._default_processor = None
			else:
				logger.warning("OpenAI Agents SDK has no default processor to restore.")
			self._processor = None
			self._exporter = None
			self._is_instrumented_instance_flag = False
			logger.info("Successfully removed OpenAI Agents SDK instrumentation")
		except Exception as e:
			logger.warning(f"Failed to uninstrument OpenAI Agents SDK: {e}", exc_info=True)


