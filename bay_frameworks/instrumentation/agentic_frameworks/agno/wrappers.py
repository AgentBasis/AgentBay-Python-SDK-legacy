from typing import Any, Optional
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Status, StatusCode

from MYSDK.bay_frameworks.instrumentation.agentic_frameworks.agno.attributes import (
	get_agent_run_attributes,
	get_metrics_attributes,
	get_team_run_attributes,
	get_tool_execution_attributes,
	get_workflow_run_attributes,
)


class StreamingResultWrapper:
	def __init__(self, original_result, span, agent_id, agent_context, streaming_context_manager):
		self.original_result = original_result
		self.span = span
		self.agent_id = agent_id
		self.agent_context = agent_context
		self.streaming_context_manager = streaming_context_manager
		self._consumed = False

	def __iter__(self):
		context_token = otel_context.attach(self.agent_context)
		try:
			for item in self.original_result:
				yield item
		finally:
			otel_context.detach(context_token)
			if not self._consumed:
				self._consumed = True
				self.span.end()
				self.streaming_context_manager.remove_context(self.agent_id)

	def __getattr__(self, name):
		return getattr(self.original_result, name)


def create_streaming_workflow_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		workflow_id = str(getattr(instance, "workflow_id", None) or getattr(instance, "id", None) or id(instance))
		workflow_name = getattr(instance, "name", None) or type(instance).__name__
		span_name = f"{workflow_name}.agno.workflow.run.workflow" if workflow_name else "agno.workflow.run.workflow"
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		if is_streaming:
			span = tracer.start_span(span_name)
			try:
				attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(workflow_id, current_context, span)
				context_token = otel_context.attach(current_context)
				try:
					result = wrapped(*args, **kwargs)
				finally:
					otel_context.detach(context_token)
				result_attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				span.set_status(Status(StatusCode.OK))
				return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(workflow_id)
				raise
		else:
			with tracer.start_as_current_span(span_name) as span:
				try:
					attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = wrapped(*args, **kwargs)
					result_attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
				except Exception as e:
					span.set_status(Status(StatusCode.ERROR, str(e)))
					span.record_exception(e)
					raise
	return wrapper


def create_streaming_workflow_async_wrapper(tracer, streaming_context_manager):
	async def wrapper(wrapped, instance, args, kwargs):
		workflow_id = str(getattr(instance, "workflow_id", None) or getattr(instance, "id", None) or id(instance))
		workflow_name = getattr(instance, "name", None) or type(instance).__name__
		span_name = f"{workflow_name}.agno.workflow.run.workflow" if workflow_name else "agno.workflow.run.workflow"
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		if is_streaming:
			span = tracer.start_span(span_name)
			try:
				attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(workflow_id, current_context, span)
				context_token = otel_context.attach(current_context)
				try:
					result = await wrapped(*args, **kwargs)
				finally:
					otel_context.detach(context_token)
				result_attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				span.set_status(Status(StatusCode.OK))
				return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(workflow_id)
				raise
		else:
			with tracer.start_as_current_span(span_name) as span:
				try:
					attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = await wrapped(*args, **kwargs)
					result_attributes = get_workflow_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
				except Exception as e:
					span.set_status(Status(StatusCode.ERROR, str(e)))
					span.record_exception(e)
					raise
	return wrapper


def create_streaming_agent_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		agent_id = str(getattr(instance, "agent_id", None) or getattr(instance, "id", None) or id(instance))
		session_id = getattr(instance, "session_id", None)
		agent_name = getattr(instance, "name", None)
		span_name = f"{agent_name}.agno.agent.run.agent" if agent_name else "agno.agent.run.agent"
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		if is_streaming:
			span = tracer.start_span(span_name)
			try:
				attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(agent_id, current_context, span)
				if session_id:
					streaming_context_manager.store_agent_session_mapping(session_id, agent_id)
				context_token = otel_context.attach(current_context)
				try:
					result = wrapped(*args, **kwargs)
				finally:
					otel_context.detach(context_token)
				result_attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				span.set_status(Status(StatusCode.OK))
				if hasattr(result, "__iter__"):
					return StreamingResultWrapper(result, span, agent_id, current_context, streaming_context_manager)
				else:
					span.end()
					streaming_context_manager.remove_context(agent_id)
					return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(agent_id)
				raise
		else:
			with tracer.start_as_current_span(span_name) as span:
				try:
					attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = wrapped(*args, **kwargs)
					result_attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
				except Exception as e:
					span.set_status(Status(StatusCode.ERROR, str(e)))
					span.record_exception(e)
					raise
	return wrapper


def create_streaming_agent_async_wrapper(tracer, streaming_context_manager):
	async def wrapper(wrapped, instance, args, kwargs):
		agent_id = str(getattr(instance, "agent_id", None) or getattr(instance, "id", None) or id(instance))
		session_id = getattr(instance, "session_id", None)
		agent_name = getattr(instance, "name", None)
		span_name = f"{agent_name}.agno.agent.run.agent" if agent_name else "agno.agent.run.agent"
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		if is_streaming:
			span = tracer.start_span(span_name)
			try:
				attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(agent_id, current_context, span)
				if session_id:
					streaming_context_manager.store_agent_session_mapping(session_id, agent_id)
				context_token = otel_context.attach(current_context)
				try:
					result = await wrapped(*args, **kwargs)
				finally:
					otel_context.detach(context_token)
				result_attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				span.set_status(Status(StatusCode.OK))
				if hasattr(result, "__iter__"):
					return StreamingResultWrapper(result, span, agent_id, current_context, streaming_context_manager)
				else:
					span.end()
					streaming_context_manager.remove_context(agent_id)
					return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(agent_id)
				raise
		else:
			with tracer.start_as_current_span(span_name) as span:
				try:
					attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = await wrapped(*args, **kwargs)
					result_attributes = get_agent_run_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
				except Exception as e:
					span.set_status(Status(StatusCode.ERROR, str(e)))
					span.record_exception(e)
					raise
	return wrapper


def create_streaming_tool_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		parent_context = None
		parent_span = None
		try:
			if hasattr(instance, "_agent"):
				agent = instance._agent
				agent_id = str(getattr(agent, "agent_id", None) or getattr(agent, "id", None) or id(agent))
				context_info = streaming_context_manager.get_context(agent_id)
				if context_info:
					parent_context, parent_span = context_info
		except Exception:
			pass
		if not parent_context:
			try:
				if hasattr(instance, "_workflow"):
					workflow = instance._workflow
					workflow_id = str(getattr(workflow, "workflow_id", None) or getattr(workflow, "id", None) or id(workflow))
					context_info = streaming_context_manager.get_context(workflow_id)
					if context_info:
						parent_context, parent_span = context_info
			except Exception:
				pass
		if parent_context:
			context_token = otel_context.attach(parent_context)
			try:
				with tracer.start_as_current_span("agno.tool.execute.tool_usage") as span:
					attributes = get_tool_execution_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = wrapped(*args, **kwargs)
					result_attributes = get_tool_execution_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				raise
			finally:
				otel_context.detach(context_token)
		else:
			with tracer.start_as_current_span("agno.tool.execute.tool_usage") as span:
				try:
					attributes = get_tool_execution_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = wrapped(*args, **kwargs)
					result_attributes = get_tool_execution_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
					for k, v in result_attributes.items():
						if k not in attributes:
							span.set_attribute(k, v)
					span.set_status(Status(StatusCode.OK))
					return result
				except Exception as e:
					span.set_status(Status(StatusCode.ERROR, str(e)))
					span.record_exception(e)
					raise
	return wrapper


def create_metrics_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		span_name = "agno.agent.metrics"
		if hasattr(instance, "model") and instance.model and hasattr(instance.model, "id"):
			model_id = str(instance.model.id)
			span_name = f"{model_id}.llm"
		with tracer.start_as_current_span(span_name) as span:
			try:
				attributes = get_metrics_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				span.set_status(Status(StatusCode.OK))
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				raise
	return wrapper


def create_team_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		team_id = str(getattr(instance, "team_id", None) or getattr(instance, "id", None) or id(instance))
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		team_name = getattr(instance, "name", None)
		base_span_name = f"{team_name}.agno.team.run.workflow" if team_name else "agno.team.run.workflow"
		if wrapped.__name__ == "print_response":
			span = tracer.start_span(base_span_name)
			try:
				attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(team_id, current_context, span)
				result = wrapped(*args, **kwargs)
				return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(team_id)
				raise
		else:
			span = tracer.start_span(base_span_name)
			try:
				attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				current_context = trace.set_span_in_context(span, otel_context.get_current())
				streaming_context_manager.store_context(team_id, current_context, span)
				context_token = otel_context.attach(current_context)
				try:
					result = wrapped(*args, **kwargs)
					if is_streaming and hasattr(result, "__iter__"):
						return StreamingResultWrapper(result, span, team_id, current_context, streaming_context_manager)
					else:
						span.end()
						streaming_context_manager.remove_context(team_id)
						return result
				finally:
					otel_context.detach(context_token)
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				span.end()
				streaming_context_manager.remove_context(team_id)
				raise
	return wrapper


def create_team_async_wrapper(tracer, streaming_context_manager):
	async def wrapper(wrapped, instance, args, kwargs):
		team_id = str(getattr(instance, "team_id", None) or getattr(instance, "id", None) or id(instance))
		is_streaming = kwargs.get("stream", getattr(instance, "stream", False))
		team_name = getattr(instance, "name", None)
		span_name = f"{team_name}.agno.team.run.workflow" if team_name else "agno.team.run.workflow"
		span = tracer.start_span(span_name)
		try:
			attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
			for k, v in attributes.items():
				span.set_attribute(k, v)
			current_context = trace.set_span_in_context(span, otel_context.get_current())
			streaming_context_manager.store_context(team_id, current_context, span)
			context_token = otel_context.attach(current_context)
			try:
				result = await wrapped(*args, **kwargs)
				if not is_streaming:
					span.end()
					streaming_context_manager.remove_context(team_id)
				return result
			finally:
				otel_context.detach(context_token)
		except Exception as e:
			span.set_status(Status(StatusCode.ERROR, str(e)))
			span.record_exception(e)
			span.end()
			streaming_context_manager.remove_context(team_id)
			raise
	return wrapper


def create_team_internal_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		team_id = str(getattr(instance, "team_id", None) or getattr(instance, "id", None) or id(instance))
		existing_context = streaming_context_manager.get_context(team_id)
		if existing_context:
			parent_context, parent_span = existing_context
			context_token = otel_context.attach(parent_context)
			try:
				with tracer.start_as_current_span("agno.team.run.workflow") as span:
					attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = wrapped(*args, **kwargs)
					span.set_status(Status(StatusCode.OK))
					return result
				finally:
					if parent_span:
						parent_span.end()
						streaming_context_manager.remove_context(team_id)
					otel_context.detach(context_token)
		else:
			with tracer.start_as_current_span("agno.team.run.workflow") as span:
				attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				result = wrapped(*args, **kwargs)
				span.set_status(Status(StatusCode.OK))
				return result
	return wrapper


def create_team_internal_async_wrapper(tracer, streaming_context_manager):
	async def wrapper(wrapped, instance, args, kwargs):
		team_id = str(getattr(instance, "team_id", None) or getattr(instance, "id", None) or id(instance))
		existing_context = streaming_context_manager.get_context(team_id)
		if existing_context:
			parent_context, parent_span = existing_context
			context_token = otel_context.attach(parent_context)
			try:
				with tracer.start_as_current_span("agno.team.run.workflow") as span:
					attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
					for k, v in attributes.items():
						span.set_attribute(k, v)
					result = await wrapped(*args, **kwargs)
					span.set_status(Status(StatusCode.OK))
					return result
				finally:
					if parent_span:
						parent_span.end()
						streaming_context_manager.remove_context(team_id)
					otel_context.detach(context_token)
		else:
			with tracer.start_as_current_span("agno.team.run.workflow") as span:
				attributes = get_team_run_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				result = await wrapped(*args, **kwargs)
				span.set_status(Status(StatusCode.OK))
				return result
	return wrapper


def create_storage_read_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		with tracer.start_as_current_span("agno.workflow.storage.read") as span:
			try:
				attributes = get_storage_read_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				result = wrapped(*args, **kwargs)
				result_attributes = get_storage_read_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				if hasattr(instance, "session_state") and isinstance(instance.session_state, dict):
					cache_size = len(instance.session_state)
					span.update_name(f"Storage.Read.{ 'Hit' if result is not None else 'Miss'}[cache:{cache_size}]")
				span.set_status(Status(StatusCode.OK))
				return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				raise
	return wrapper


def create_storage_write_wrapper(tracer, streaming_context_manager):
	def wrapper(wrapped, instance, args, kwargs):
		with tracer.start_as_current_span("agno.workflow.storage.write") as span:
			try:
				attributes = get_storage_write_attributes(args=(instance,) + args, kwargs=kwargs)
				for k, v in attributes.items():
					span.set_attribute(k, v)
				result = wrapped(*args, **kwargs)
				result_attributes = get_storage_write_attributes(args=(instance,) + args, kwargs=kwargs, return_value=result)
				for k, v in result_attributes.items():
					if k not in attributes:
						span.set_attribute(k, v)
				if hasattr(instance, "session_state") and isinstance(instance.session_state, dict):
					cache_size = len(instance.session_state)
					span.update_name(f"Storage.Write[cache:{cache_size}]")
				span.set_status(Status(StatusCode.OK))
				return result
			except Exception as e:
				span.set_status(Status(StatusCode.ERROR, str(e)))
				span.record_exception(e)
				raise
	return wrapper


def create_workflow_init_wrapper(tracer):
	def wrapper(wrapped, instance, args, kwargs):
		result = wrapped(*args, **kwargs)
		if hasattr(instance, "session_state") and isinstance(instance.session_state, dict):
			original_state = instance.session_state
			instance.session_state = original_state
		return result
	return wrapper


