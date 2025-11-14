from typing import Any, Dict, Optional, Tuple
import json

from agentbay.bay_frameworks.instrumentation.common.attributes import get_common_attributes
from agentbay.bay_frameworks.semconv.message import MessageAttributes
from agentbay.bay_frameworks.semconv.span_attributes import SpanAttributes


def get_model_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		model_name = "unknown"
		if kwargs:
			if "model" in kwargs:
				model_name = kwargs["model"]
			elif kwargs.get("self") and hasattr(kwargs["self"], "model_id"):
				model_name = kwargs["self"].model_id
		if model_name == "unknown" and args and len(args) > 0:
			instance = args[0]
			if hasattr(instance, "model_id"):
				model_name = instance.model_id
		attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
		if kwargs and "messages" in kwargs:
			messages = kwargs["messages"]
			if isinstance(messages, list):
				for i, message in enumerate(messages):
					message_dict = message
					if hasattr(message, "to_dict"):
						message_dict = message.to_dict()
					elif hasattr(message, "__dict__"):
						message_dict = message.__dict__
					if isinstance(message_dict, dict):
						role = message_dict.get("role", "user")
						attributes[MessageAttributes.PROMPT_ROLE.format(i=i)] = role
						content = message_dict.get("content", "")
						if content:
							attributes[MessageAttributes.PROMPT_CONTENT.format(i=i)] = str(content)
		if kwargs and "tools_to_call_from" in kwargs:
			tools = kwargs["tools_to_call_from"]
			if tools and isinstance(tools, list):
				for i, tool in enumerate(tools):
					tool_name = getattr(tool, "name", "unknown")
					tool_description = getattr(tool, "description", "")
					attributes[MessageAttributes.TOOL_CALL_NAME.format(i=i)] = tool_name
					if tool_description:
						attributes[MessageAttributes.TOOL_CALL_DESCRIPTION.format(i=i)] = tool_description
		if kwargs:
			if "temperature" in kwargs:
				attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] = kwargs["temperature"]
			if "max_tokens" in kwargs:
				attributes[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = kwargs["max_tokens"]
			if "stop_sequences" in kwargs:
				attributes[SpanAttributes.LLM_REQUEST_STOP_SEQUENCES] = json.dumps(kwargs["stop_sequences"])
		if return_value:
			try:
				if hasattr(return_value, "content"):
					attributes[MessageAttributes.COMPLETION_CONTENT.format(i=0)] = str(return_value.content)
				if hasattr(return_value, "role"):
					attributes[MessageAttributes.COMPLETION_ROLE.format(i=0)] = return_value.role
				if hasattr(return_value, "tool_calls") and return_value.tool_calls:
					for j, tool_call in enumerate(return_value.tool_calls):
						if hasattr(tool_call, "function"):
							attributes[MessageAttributes.COMPLETION_TOOL_CALL_NAME.format(i=0, j=j)] = tool_call.function.name
							if hasattr(tool_call.function, "arguments"):
								attributes[MessageAttributes.COMPLETION_TOOL_CALL_ARGUMENTS.format(i=0, j=j)] = (
									ool_call.function.arguments
								)
							if hasattr(tool_call, "id"):
								attributes[MessageAttributes.COMPLETION_TOOL_CALL_ID.format(i=0, j=j)] = tool_call.id
				if hasattr(return_value, "token_usage") and return_value.token_usage:
					tu = return_value.token_usage
					if hasattr(tu, "input_tokens"):
						attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = tu.input_tokens
					if hasattr(tu, "output_tokens"):
						attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = tu.output_tokens
					if hasattr(tu, "total_tokens"):
						attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = tu.total_tokens
				if hasattr(return_value, "raw") and return_value.raw:
					raw_response = return_value.raw
					if hasattr(raw_response, "id"):
						attributes[SpanAttributes.LLM_RESPONSE_ID] = raw_response.id
		except Exception:
			pass
	except Exception:
		pass
	return attributes


def get_stream_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		model_name = "unknown"
		if kwargs and kwargs.get("self") and hasattr(kwargs["self"], "model_id"):
			model_name = kwargs["self"].model_id
		elif args and len(args) > 0 and hasattr(args[0], "model_id"):
			model_name = args[0].model_id
		attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
		attributes["gen_ai.request.streaming"] = True
		if kwargs and "messages" in kwargs:
			messages = kwargs["messages"]
			if isinstance(messages, list):
				for i, message in enumerate(messages):
					message_dict = message
					if hasattr(message, "to_dict"):
						message_dict = message.to_dict()
					elif hasattr(message, "__dict__"):
						message_dict = message.__dict__
					if isinstance(message_dict, dict):
						role = message_dict.get("role", "user")
						attributes[MessageAttributes.PROMPT_ROLE.format(i=i)] = role
						content = message_dict.get("content", "")
						if content:
							attributes[MessageAttributes.PROMPT_CONTENT.format(i=i)] = str(content)
	except Exception:
		pass
	return attributes


