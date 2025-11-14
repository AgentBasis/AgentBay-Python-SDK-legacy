from typing import Dict, Any
from opentelemetry.trace import SpanKind
from opentelemetry.metrics import Meter
from wrapt import wrap_function_wrapper

from agentbay.bay_frameworks.instrumentation.common import CommonInstrumentor, StandardMetrics, InstrumentorConfig
from agentbay.bay_frameworks.logging import logger


LIBRARY_NAME = "bay_frameworks.instrumentation.smolagents"
LIBRARY_VERSION = "0.1.0"

try:
	from agentbay.bay_frameworks.instrumentation.agentic_frameworks.smolagents.attributes.agent import (
		get_agent_attributes,
		get_tool_call_attributes,
		get_planning_step_attributes,
		get_agent_step_attributes,
		get_agent_stream_attributes,
		get_managed_agent_attributes,
	)
	from agentbay.bay_frameworks.instrumentation.agentic_frameworks.smolagents.attributes.model import (
		get_model_attributes,
		get_stream_attributes,
	)
except ImportError:
	def get_agent_attributes(*args, **kwargs):
		return {}

	def get_tool_call_attributes(*args, **kwargs):
		return {}

	def get_planning_step_attributes(*args, **kwargs):
		return {}

	def get_agent_step_attributes(*args, **kwargs):
		return {}

	def get_agent_stream_attributes(*args, **kwargs):
		return {}

	def get_managed_agent_attributes(*args, **kwargs):
		return {}

	def get_model_attributes(*args, **kwargs):
		return {}

	def get_stream_attributes(*args, **kwargs):
		return {}


class SmolagentsInstrumentor(CommonInstrumentor):
	"""Instrumentor for SmoLAgents library."""

	def __init__(self):
		config = InstrumentorConfig(
			library_name=LIBRARY_NAME,
			library_version=LIBRARY_VERSION,
			wrapped_methods=[],
			metrics_enabled=True,
			dependencies=["smolagents >= 1.0.0", "litellm"],
		)
		super().__init__(config)

	def _create_metrics(self, meter: Meter) -> Dict[str, Any]:
		return StandardMetrics.create_standard_metrics(meter)

	def _custom_wrap(self, **kwargs):
		wrap_function_wrapper("smolagents.agents", "CodeAgent.run", self._agent_run_wrapper(self._tracer))
		wrap_function_wrapper("smolagents.agents", "ToolCallingAgent.run", self._agent_run_wrapper(self._tracer))
		wrap_function_wrapper(
			"smolagents.agents", "ToolCallingAgent.execute_tool_call", self._tool_execution_wrapper(self._tracer)
		)
		wrap_function_wrapper("smolagents.models", "LiteLLMModel.generate", self._llm_wrapper(self._tracer))
		wrap_function_wrapper("smolagents.models", "LiteLLMModel.generate_stream", self._llm_wrapper(self._tracer))
		logger.info("SmoLAgents instrumentation enabled")

	def _agent_run_wrapper(self, tracer):
		def wrapper(wrapped, instance, args, kwargs):
			agent_name = getattr(instance, "name", None) or instance.__class__.__name__
			span_name = f"{agent_name}.run"
			with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
				attributes = get_agent_attributes(args=(instance,) + args, kwargs=kwargs)
				if hasattr(instance, "managed_agents") and instance.managed_agents:
					managed_agent_names = []
					for agent in instance.managed_agents:
						name = getattr(agent, "name", None) or agent.__class__.__name__
						managed_agent_names.append(name)
					attributes["agent.managed_agents"] = str(managed_agent_names)
				else:
					attributes["agent.managed_agents"] = "[]"
				for key, value in attributes.items():
					if value is not None:
						span.set_attribute(key, value)
				try:
					result = wrapped(*args, **kwargs)
					if result is not None:
						span.set_attribute("bay_frameworks.entity.output", str(result))
					return result
				except Exception as e:
					span.record_exception(e)
					raise
		return wrapper

	def _tool_execution_wrapper(self, tracer):
		def wrapper(wrapped, instance, args, kwargs):
			tool_name = "unknown"
			if args and len(args) > 0:
				tool_call = args[0]
				if hasattr(tool_call, "function"):
					tool_name = tool_call.function.name
			span_name = f"tool.{tool_name}" if tool_name != "unknown" else "tool.execute"
			with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
				tool_params = "{}"
				if args and len(args) > 0:
					tool_call = args[0]
					if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
						tool_params = str(tool_call.function.arguments)
				attributes = get_tool_call_attributes(args=(instance,) + args, kwargs=kwargs)
				if tool_name != "unknown":
					attributes["tool.name"] = tool_name
					attributes["tool.parameters"] = tool_params
				for key, value in attributes.items():
					if value is not None:
						span.set_attribute(key, value)
				try:
					result = wrapped(*args, **kwargs)
					span.set_attribute("tool.status", "success")
					if result is not None:
						span.set_attribute("tool.result", str(result))
					return result
				except Exception as e:
					span.set_attribute("tool.status", "error")
					span.record_exception(e)
					raise
		return wrapper

	def _llm_wrapper(self, tracer):
		def wrapper(wrapped, instance, args, kwargs):
			model_name = getattr(instance, "model_id", "unknown")
			is_streaming = "generate_stream" in wrapped.__name__
			operation = "generate_stream" if is_streaming else "generate"
			span_name = f"litellm.{operation} ({model_name})" if model_name != "unknown" else f"litellm.{operation}"
			with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
				attributes = get_stream_attributes(args=(instance,) + args, kwargs=kwargs) if is_streaming else get_model_attributes(args=(instance,) + args, kwargs=kwargs)
				attributes["gen_ai.request.model"] = model_name
				for key, value in attributes.items():
					if value is not None:
						span.set_attribute(key, value)
				try:
					result = wrapped(*args, **kwargs)
					if result and hasattr(result, "content"):
						span.set_attribute("gen_ai.response.0.content", str(result.content))
					if result and hasattr(result, "token_usage"):
						tu = result.token_usage
						if hasattr(tu, "input_tokens"):
							span.set_attribute("gen_ai.usage.prompt_tokens", tu.input_tokens)
						if hasattr(tu, "output_tokens"):
							span.set_attribute("gen_ai.usage.completion_tokens", tu.output_tokens)
						if hasattr(tu, "total_tokens"):
							span.set_attribute("gen_ai.usage.total_tokens", tu.total_tokens)
					if hasattr(result, "raw") and result.raw and hasattr(result.raw, "id"):
						span.set_attribute("gen_ai.response.id", result.raw.id)
					return result
				except Exception as e:
					span.record_exception(e)
					raise
		return wrapper


