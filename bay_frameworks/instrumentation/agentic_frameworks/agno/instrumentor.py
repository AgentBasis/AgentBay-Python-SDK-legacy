from typing import List, Any, Optional, Dict
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Meter
import threading

from agentbay.bay_frameworks.logging import logger
from agentbay.bay_frameworks.instrumentation.common import (
	CommonInstrumentor,
	StandardMetrics,
	InstrumentorConfig,
)
from agentbay.bay_frameworks.instrumentation.common.wrappers import WrapConfig, wrap_function_wrapper, wrap

from agentbay.bay_frameworks.instrumentation.agentic_frameworks.agno.attributes import (
	get_agent_run_attributes,
	get_metrics_attributes,
	get_team_run_attributes,
	get_tool_execution_attributes,
	get_workflow_run_attributes,
	get_workflow_session_attributes,
	get_storage_read_attributes,
	get_storage_write_attributes,
)


WRAPPED_METHODS: List[WrapConfig] = []


class StreamingContextManager:
	def __init__(self):
		self._contexts = {}
		self._agent_sessions = {}
		self._lock = threading.Lock()

	def store_context(self, context_id: str, span_context: Any, span: Any) -> None:
		with self._lock:
			self._contexts[context_id] = (span_context, span)

	def get_context(self, context_id: str) -> Optional[tuple]:
		with self._lock:
			return self._contexts.get(context_id)

	def remove_context(self, context_id: str) -> None:
		with self._lock:
			self._contexts.pop(context_id, None)

	def store_agent_session_mapping(self, session_id: str, agent_id: str) -> None:
		with self._lock:
			self._agent_sessions[session_id] = agent_id

	def get_agent_context_by_session(self, session_id: str) -> Optional[tuple]:
		with self._lock:
			agent_id = self._agent_sessions.get(session_id)
			if agent_id:
				return self._contexts.get(agent_id)
			return None

	def clear_all(self) -> None:
		with self._lock:
			self._contexts.clear()
			self._agent_sessions.clear()


class AgnoInstrumentor(CommonInstrumentor):
	def __init__(self):
		self._streaming_context_manager = StreamingContextManager()
		config = InstrumentorConfig(
			library_name="bay_frameworks.instrumentation.agno",
			library_version="0.1.0",
			wrapped_methods=WRAPPED_METHODS.copy(),
			metrics_enabled=True,
			dependencies=["agno >= 0.1.0"],
		)
		super().__init__(config)

	def _create_metrics(self, meter: Meter) -> Dict[str, Any]:
		return StandardMetrics.create_standard_metrics(meter)

	def _initialize(self, **kwargs):
		logger.info("Agno instrumentation: Beginning immediate instrumentation")
		try:
			self._perform_wrapping()
			logger.info("Agno instrumentation: Immediate instrumentation completed successfully")
		except Exception as e:
			logger.error(f"Failed to perform immediate wrapping: {e}")

	def _custom_wrap(self, **kwargs):
		pass

	def _perform_wrapping(self):
		if not self._tracer:
			logger.debug("No tracer available for Agno wrapping")
			return
		try:
			import agno.agent
			import agno.workflow.workflow
			import agno.tools.function
			import agno.team.team  # Noqa: F401
		except ImportError as e:
			logger.error(f"Failed to import Agno modules for wrapping: {e}")
			return
		# Session wrappers
		for wc in [
			WrapConfig(
				trace_name="agno.workflow.session.load_session",
				package="agno.workflow.workflow",
				class_name="Workflow",
				method_name="load_session",
				handler=get_workflow_session_attributes,
			),
			WrapConfig(
				trace_name="agno.workflow.session.new_session",
				package="agno.workflow.workflow",
				class_name="Workflow",
				method_name="new_session",
				handler=get_workflow_session_attributes,
			),
		]:
			try:
				wrap(wc, self._tracer)
			except Exception as e:
				logger.debug(f"Failed to wrap {wc}: {e}")
		# Streaming and team wrappers
		wrap_function_wrapper("agno.agent", "Agent.run", self._create_streaming_agent_wrapper())
		wrap_function_wrapper("agno.agent", "Agent.arun", self._create_streaming_agent_async_wrapper())
		wrap_function_wrapper("agno.workflow.workflow", "Workflow.run_workflow", self._create_streaming_workflow_wrapper())
		wrap_function_wrapper(
			"agno.workflow.workflow", "Workflow.arun_workflow", self._create_streaming_workflow_async_wrapper()
		)
		wrap_function_wrapper("agno.tools.function", "FunctionCall.execute", self._create_streaming_tool_wrapper())
		wrap_function_wrapper("agno.agent", "Agent._set_session_metrics", self._create_metrics_wrapper())
		wrap_function_wrapper("agno.team.team", "Team.print_response", self._create_team_wrapper())
		wrap_function_wrapper("agno.team.team", "Team.run", self._create_team_wrapper())
		wrap_function_wrapper("agno.team.team", "Team.arun", self._create_team_async_wrapper())
		wrap_function_wrapper("agno.team.team", "Team._run", self._create_team_internal_wrapper())
		wrap_function_wrapper("agno.team.team", "Team._arun", self._create_team_internal_async_wrapper())
		wrap_function_wrapper("agno.workflow.workflow", "Workflow.read_from_storage", self._create_storage_read_wrapper())
		wrap_function_wrapper("agno.workflow.workflow", "Workflow.write_to_storage", self._create_storage_write_wrapper())
		wrap_function_wrapper("agno.workflow.workflow", "Workflow.__init__", self._create_workflow_init_wrapper())

	def _custom_unwrap(self, **kwargs):
		self._streaming_context_manager.clear_all()
		logger.info("Agno instrumentation removed successfully")

	# instance wrapper factories
	def _create_streaming_agent_wrapper(self, *_, **__):
		return create_streaming_agent_wrapper(self._tracer, self._streaming_context_manager)

	def _create_streaming_agent_async_wrapper(self, *_, **__):
		return create_streaming_agent_async_wrapper(self._tracer, self._streaming_context_manager)

	def _create_streaming_workflow_wrapper(self, *_, **__):
		return create_streaming_workflow_wrapper(self._tracer, self._streaming_context_manager)

	def _create_streaming_workflow_async_wrapper(self, *_, **__):
		return create_streaming_workflow_async_wrapper(self._tracer, self._streaming_context_manager)

	def _create_streaming_tool_wrapper(self, *_, **__):
		return create_streaming_tool_wrapper(self._tracer, self._streaming_context_manager)

	def _create_metrics_wrapper(self, *_, **__):
		return create_metrics_wrapper(self._tracer, self._streaming_context_manager)

	def _create_team_wrapper(self, *_, **__):
		return create_team_wrapper(self._tracer, self._streaming_context_manager)

	def _create_team_async_wrapper(self, *_, **__):
		return create_team_async_wrapper(self._tracer, self._streaming_context_manager)

	def _create_team_internal_wrapper(self, *_, **__):
		return create_team_internal_wrapper(self._tracer, self._streaming_context_manager)

	def _create_team_internal_async_wrapper(self, *_, **__):
		return create_team_internal_async_wrapper(self._tracer, self._streaming_context_manager)

	def _create_storage_read_wrapper(self, *_, **__):
		return create_storage_read_wrapper(self._tracer, self._streaming_context_manager)

	def _create_storage_write_wrapper(self, *_, **__):
		return create_storage_write_wrapper(self._tracer, self._streaming_context_manager)

	def _create_workflow_init_wrapper(self, *_, **__):
		return create_workflow_init_wrapper(self._tracer)


# Import wrapper builders from the original module (ported and rebranded below)
from agentbay.bay_frameworks.instrumentation.agentic_frameworks.agno.wrappers import (
	create_streaming_agent_wrapper,
	create_streaming_agent_async_wrapper,
	create_streaming_workflow_wrapper,
	create_streaming_workflow_async_wrapper,
	create_streaming_tool_wrapper,
	create_metrics_wrapper,
	create_team_wrapper,
	create_team_async_wrapper,
	create_team_internal_wrapper,
	create_team_internal_async_wrapper,
	create_storage_read_wrapper,
	create_storage_write_wrapper,
	create_workflow_init_wrapper,
)


