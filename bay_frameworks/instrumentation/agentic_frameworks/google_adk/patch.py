import json
import wrapt
from typing import Any
from opentelemetry import trace as opentelemetry_api_trace
from opentelemetry.trace import SpanKind as SpanKind

from MYSDK.bay_frameworks.logging import logger
from MYSDK.bay_frameworks.semconv import SpanAttributes, ToolAttributes, MessageAttributes, AgentAttributes


_wrapped_methods = []


class NoOpSpan:
	def __enter__(self):
		return self

	def __exit__(self, *args):
		pass

	def set_attribute(self, *args, **kwargs):
		pass

	def set_attributes(self, *args, **kwargs):
		pass

	def add_event(self, *args, **kwargs):
		pass

	def set_status(self, *args, **kwargs):
		pass

	def update_name(self, *args, **kwargs):
		pass

	def is_recording(self):
		return False

	def end(self, *args, **kwargs):
		pass

	def record_exception(self, *args, **kwargs):
		pass


class NoOpTracer:
	def start_as_current_span(self, *args, **kwargs):
		return NoOpSpan()

	def start_span(self, *args, **kwargs):
		return NoOpSpan()

	def use_span(self, *args, **kwargs):
		return NoOpSpan()


def _build_llm_request_for_trace(llm_request) -> dict:
	from google.genai import types
	result = {"model": llm_request.model, "config": llm_request.config.model_dump(exclude_none=True, exclude="response_schema"), "contents": []}
	for content in llm_request.contents:
		parts = [part for part in content.parts if not hasattr(part, "inline_data") or not part.inline_data]
		result["contents"].append(types.Content(role=content.role, parts=parts).model_dump(exclude_none=True))
	return result


def _extract_messages_from_contents(contents: list) -> dict:
	attributes = {}
	for i, content in enumerate(contents):
		raw_role = content.get("role", "user")
		if raw_role == "model":
			role = "assistant"
		elif raw_role == "user":
			role = "user"
		elif raw_role == "system":
			role = "system"
		else:
			role = raw_role
		parts = content.get("parts", [])
		attributes[MessageAttributes.PROMPT_ROLE.format(i=i)] = role
		text_parts = []
		for part in parts:
			if "text" in part and part.get("text") is not None:
				text_parts.append(str(part["text"]))
			elif "function_call" in part:
				func_call = part["function_call"]
				attributes[f"gen_ai.prompt.{i}.function_call.name"] = func_call.get("name", "")
				attributes[f"gen_ai.prompt.{i}.function_call.args"] = json.dumps(func_call.get("args", {}))
				if "id" in func_call:
					attributes[f"gen_ai.prompt.{i}.function_call.id"] = func_call["id"]
			elif "function_response" in part:
				func_resp = part["function_response"]
				attributes[f"gen_ai.prompt.{i}.function_response.name"] = func_resp.get("name", "")
				attributes[f"gen_ai.prompt.{i}.function_response.result"] = json.dumps(func_resp.get("response", {}))
				if "id" in func_resp:
					attributes[f"gen_ai.prompt.{i}.function_response.id"] = func_resp["id"]
		if text_parts:
			attributes[MessageAttributes.PROMPT_CONTENT.format(i=i)] = "\n".join(text_parts)
	return attributes


def _extract_llm_attributes(llm_request_dict: dict, llm_response: Any) -> dict:
	attributes = {}
	if "model" in llm_request_dict:
		attributes[SpanAttributes.LLM_REQUEST_MODEL] = llm_request_dict["model"]
	if "config" in llm_request_dict:
		config = llm_request_dict["config"]
		if "temperature" in config:
			attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = config["temperature"]
		if "max_output_tokens" in config:
			attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = config["max_output_tokens"]
		if "top_p" in config:
			attributes[SpanAttributes.LLM_REQUEST_TOP_P] = config["top_p"]
		if "top_k" in config:
			attributes[SpanAttributes.LLM_REQUEST_TOP_K] = config["top_k"]
		if "candidate_count" in config:
			attributes[SpanAttributes.LLM_REQUEST_CANDIDATE_COUNT] = config["candidate_count"]
		if "stop_sequences" in config:
			attributes[SpanAttributes.LLM_REQUEST_STOP_SEQUENCES] = json.dumps(config["stop_sequences"])
		if "response_mime_type" in config:
			attributes["gen_ai.request.response_mime_type"] = config["response_mime_type"]
		if "tools" in config:
			for i, tool in enumerate(config["tools"]):
				if "function_declarations" in tool:
					for j, func in enumerate(tool["function_declarations"]):
						attributes[f"gen_ai.request.tools.{j}.name"] = func.get("name", "")
						attributes[f"gen_ai.request.tools.{j}.description"] = func.get("description", "")
	message_index = 0
	if "config" in llm_request_dict and "system_instruction" in llm_request_dict["config"]:
		system_instruction = llm_request_dict["config"]["system_instruction"]
		attributes[MessageAttributes.PROMPT_ROLE.format(i=message_index)] = "system"
		attributes[MessageAttributes.PROMPT_CONTENT.format(i=message_index)] = system_instruction
		message_index += 1
	if "contents" in llm_request_dict:
		for content in llm_request_dict["contents"]:
			raw_role = content.get("role", "user")
			if raw_role == "model":
				role = "assistant"
			elif raw_role == "user":
				role = "user"
			elif raw_role == "system":
				role = "system"
			else:
				role = raw_role
			parts = content.get("parts", [])
			attributes[MessageAttributes.PROMPT_ROLE.format(i=message_index)] = role
			text_parts = []
			for part in parts:
				if "text" in part and part.get("text") is not None:
					text_parts.append(str(part["text"]))
				elif "function_call" in part:
					func_call = part["function_call"]
					attributes[f"gen_ai.prompt.{message_index}.function_call.name"] = func_call.get("name", "")
					attributes[f"gen_ai.prompt.{message_index}.function_call.args"] = json.dumps(func_call.get("args", {}))
					if "id" in func_call:
						attributes[f"gen_ai.prompt.{message_index}.function_call.id"] = func_call["id"]
				elif "function_response" in part:
					func_resp = part["function_response"]
					attributes[f"gen_ai.prompt.{message_index}.function_response.name"] = func_resp.get("name", "")
					attributes[f"gen_ai.prompt.{message_index}.function_response.result"] = json.dumps(func_resp.get("response", {}))
					if "id" in func_resp:
						attributes[f"gen_ai.prompt.{message_index}.function_response.id"] = func_resp["id"]
			if text_parts:
				attributes[MessageAttributes.PROMPT_CONTENT.format(i=message_index)] = "\n".join(text_parts)
			message_index += 1
	if llm_response:
		try:
			response_dict = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
			if "model" in response_dict:
				attributes[SpanAttributes.LLM_RESPONSE_MODEL] = response_dict["model"]
			if "usage_metadata" in response_dict:
				usage = response_dict["usage_metadata"]
				if "prompt_token_count" in usage:
					attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["prompt_token_count"]
				if "candidates_token_count" in usage:
					attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage["candidates_token_count"]
				if "total_token_count" in usage:
					attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total_token_count"]
				if "prompt_tokens_details" in usage:
					for detail in usage["prompt_tokens_details"]:
						if "modality" in detail and "token_count" in detail:
							attributes[f"gen_ai.usage.prompt_tokens.{detail['modality'].lower()}"] = detail["token_count"]
				if "candidates_tokens_details" in usage:
					for detail in usage["candidates_tokens_details"]:
						if "modality" in detail and "token_count" in detail:
							attributes[f"gen_ai.usage.completion_tokens.{detail['modality'].lower()}"] = detail["token_count"]
			if "content" in response_dict and "parts" in response_dict["content"]:
				parts = response_dict["content"]["parts"]
				attributes[MessageAttributes.COMPLETION_ROLE.format(i=0)] = "assistant"
				text_parts = []
				tool_call_index = 0
				for part in parts:
					if "text" in part and part.get("text") is not None:
						text_parts.append(str(part["text"]))
					elif "function_call" in part:
						func_call = part["function_call"]
						attributes[MessageAttributes.COMPLETION_TOOL_CALL_NAME.format(i=0, j=tool_call_index)] = func_call.get("name", "")
						attributes[MessageAttributes.COMPLETION_TOOL_CALL_ARGUMENTS.format(i=0, j=tool_call_index)] = json.dumps(func_call.get("args", {}))
						if "id" in func_call:
							attributes[MessageAttributes.COMPLETION_TOOL_CALL_ID.format(i=0, j=tool_call_index)] = func_call["id"]
						tool_call_index += 1
				if text_parts:
					attributes[MessageAttributes.COMPLETION_CONTENT.format(i=0)] = "\n".join(text_parts)
			if "finish_reason" in response_dict:
				attributes[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = response_dict["finish_reason"]
			if "id" in response_dict:
				attributes[SpanAttributes.LLM_RESPONSE_ID] = response_dict["id"]
		except Exception as e:
			logger.debug(f"Failed to extract response attributes: {e}")
	return attributes


def _patch(module_name: str, object_name: str, method_name: str, wrapper_function, agentbay_tracer):
	try:
		module = __import__(module_name, fromlist=[object_name])
		obj = getattr(module, object_name)
		wrapt.wrap_function_wrapper(obj, method_name, wrapper_function(agentbay_tracer))
		_wrapped_methods.append((obj, method_name))
		logger.debug(f"Successfully wrapped {module_name}.{object_name}.{method_name}")
	except Exception as e:
		logger.warning(f"Could not wrap {module_name}.{object_name}.{method_name}: {e}")


def _patch_module_function(module_name: str, function_name: str, wrapper_function, agentbay_tracer):
	try:
		module = __import__(module_name, fromlist=[function_name])
		wrapt.wrap_function_wrapper(module, function_name, wrapper_function(agentbay_tracer))
		_wrapped_methods.append((module, function_name))
		logger.debug(f"Successfully wrapped {module_name}.{function_name}")
	except Exception as e:
		logger.warning(f"Could not wrap {module_name}.{function_name}: {e}")


def patch_adk(agentbay_tracer):
	logger.debug("Applying Google ADK patches for agentbay instrumentation")
	noop_tracer = NoOpTracer()
	try:
		import google.adk.telemetry as adk_telemetry
		adk_telemetry.tracer = noop_tracer
		logger.debug("Replaced ADK's tracer with NoOpTracer")
	except Exception as e:
		logger.warning(f"Failed to replace ADK tracer: {e}")
	import sys
	modules_to_patch = [
		"google.adk.runners",
		"google.adk.agents.base_agent",
		"google.adk.flows.llm_flows.base_llm_flow",
		"google.adk.flows.llm_flows.functions",
	]
	for module_name in modules_to_patch:
		if module_name in sys.modules:
			try:
				module = sys.modules[module_name]
				if hasattr(module, "tracer"):
					module.tracer = noop_tracer
					logger.debug(f"Replaced tracer in {module_name}")
			except Exception as e:
				logger.warning(f"Failed to replace tracer in {module_name}: {e}")
		_patch("google.adk.agents.base_agent", "BaseAgent", "run_async", _base_agent_run_async_wrapper, agentbay_tracer)
		_patch_module_function("google.adk.telemetry", "trace_tool_call", _adk_trace_tool_call_wrapper, agentbay_tracer)
		_patch_module_function("google.adk.telemetry", "trace_tool_response", _adk_trace_tool_response_wrapper, agentbay_tracer)
		_patch_module_function("google.adk.telemetry", "trace_call_llm", _adk_trace_call_llm_wrapper, agentbay_tracer)
		_patch_module_function("google.adk.telemetry", "trace_send_data", _adk_trace_send_data_wrapper, agentbay_tracer)
		_patch("google.adk.flows.llm_flows.base_llm_flow", "BaseLlmFlow", "_call_llm_async", _base_llm_flow_call_llm_async_wrapper, agentbay_tracer)
		_patch("google.adk.flows.llm_flows.base_llm_flow", "BaseLlmFlow", "_finalize_model_response_event", _finalize_model_response_event_wrapper, agentbay_tracer)
		_patch_module_function("google.adk.flows.llm_flows.functions", "__call_tool_async", _call_tool_async_wrapper, agentbay_tracer)
	logger.info("Google ADK patching complete")


def unpatch_adk():
	logger.debug("Removing Google ADK patches")
	try:
		import google.adk.telemetry as adk_telemetry
		from opentelemetry import trace
		adk_telemetry.tracer = trace.get_tracer("gcp.vertex.agent")
		logger.debug("Restored ADK's built-in tracer")
	except Exception as e:
		logger.warning(f"Failed to restore ADK tracer: {e}")
	for obj, method_name in _wrapped_methods:
		try:
			if hasattr(getattr(obj, method_name), "__wrapped__"):
				original = getattr(obj, method_name).__wrapped__
				setattr(obj, method_name, original)
				logger.debug(f"Successfully unwrapped {obj}.{method_name}")
		except Exception as e:
			logger.warning(f"Failed to unwrap {obj}.{method_name}: {e}")
	_wrapped_methods.clear()
	logger.info("Google ADK unpatching complete")

