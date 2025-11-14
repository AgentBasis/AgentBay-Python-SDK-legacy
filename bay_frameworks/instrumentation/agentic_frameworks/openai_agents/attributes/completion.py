from typing import Any, Dict

from agentbay.bay_frameworks.instrumentation.common.attributes import AttributeMap
from agentbay.bay_frameworks.logging import logger
from agentbay.bay_frameworks.helpers.serialization import model_to_dict
from agentbay.bay_frameworks.semconv import (
	SpanAttributes,
	MessageAttributes,
)
from agentbay.bay_frameworks.instrumentation.agentic_frameworks.openai_agents.attributes.tokens import process_token_usage


def get_generation_output_attributes(output: Any) -> Dict[str, Any]:
	response_dict = model_to_dict(output)
	result: AttributeMap = {}
	if not response_dict:
		if isinstance(output, str):
			return {}
		return result
	if "raw_responses" in response_dict and isinstance(response_dict["raw_responses"], list):
		result.update(get_raw_response_attributes(response_dict))
	else:
		if "choices" in response_dict:
			result.update(get_chat_completions_attributes(response_dict))
		usage_attributes: AttributeMap = {}
		if "usage" in response_dict:
			process_token_usage(response_dict["usage"], usage_attributes)
			result.update(usage_attributes)
		if hasattr(output, "usage") and output.usage:
			direct_usage_attributes: AttributeMap = {}
			process_token_usage(output.usage, direct_usage_attributes)
			result.update(direct_usage_attributes)
	return result


def get_raw_response_attributes(response: Dict[str, Any]) -> Dict[str, Any]:
	result: AttributeMap = {}
	result[SpanAttributes.LLM_SYSTEM] = "openai"
	if "raw_responses" in response and isinstance(response["raw_responses"], list):
		for i, raw_response in enumerate(response["raw_responses"]):
			if "usage" in raw_response and isinstance(raw_response["usage"], dict):
				usage_attrs: AttributeMap = {}
				process_token_usage(raw_response["usage"], usage_attrs)
				result.update(usage_attrs)
				logger.debug(f"Extracted token usage from raw_responses[{i}]: {usage_attrs}")
			if "output" in raw_response and isinstance(raw_response["output"], list):
				for j, output_item in enumerate(raw_response["output"]):
					if "content" in output_item and isinstance(output_item["content"], list):
						for content_item in output_item["content"]:
							if content_item.get("type") == "output_text" and "text" in content_item:
								result[MessageAttributes.COMPLETION_CONTENT.format(i=j)] = content_item["text"]
						if "role" in output_item:
							result[MessageAttributes.COMPLETION_ROLE.format(i=j)] = output_item["role"]
						if "tool_calls" in output_item and isinstance(output_item["tool_calls"], list):
							for k, tool_call in enumerate(output_item["tool_calls"]):
								tool_id = tool_call.get("id", "")
								if "function" in tool_call and isinstance(tool_call["function"], dict):
									function = tool_call["function"]
									result[MessageAttributes.COMPLETION_TOOL_CALL_ID.format(i=j, j=k)] = tool_id
									result[MessageAttributes.COMPLETION_TOOL_CALL_NAME.format(i=j, j=k)] = function.get("name", "")
									result[MessageAttributes.COMPLETION_TOOL_CALL_ARGUMENTS.format(i=j, j=k)] = function.get("arguments", "")
	return result


def get_chat_completions_attributes(response: Dict[str, Any]) -> Dict[str, Any]:
	result: AttributeMap = {}
	if "choices" not in response:
		return result
	for i, choice in enumerate(response["choices"]):
		if "finish_reason" in choice:
			result[MessageAttributes.COMPLETION_FINISH_REASON.format(i=i)] = choice["finish_reason"]
		message = choice.get("message", {})
		if "role" in message:
			result[MessageAttributes.COMPLETION_ROLE.format(i=i)] = message["role"]
		if "content" in message:
			content = message["content"] if message["content"] is not None else ""
			result[MessageAttributes.COMPLETION_CONTENT.format(i=i)] = content
		if "tool_calls" in message and message["tool_calls"] is not None:
			tool_calls = message["tool_calls"]
			for j, tool_call in enumerate(tool_calls):
				if "function" in tool_call:
					function = tool_call["function"]
					result[MessageAttributes.COMPLETION_TOOL_CALL_ID.format(i=i, j=j)] = tool_call.get("id")
					result[MessageAttributes.COMPLETION_TOOL_CALL_NAME.format(i=i, j=j)] = function.get("name")
					result[MessageAttributes.COMPLETION_TOOL_CALL_ARGUMENTS.format(i=i, j=j)] = function.get("arguments")
		if "function_call" in message and message["function_call"] is not None:
			function_call = message["function_call"]
			result[MessageAttributes.COMPLETION_TOOL_CALL_NAME.format(i=i)] = function_call.get("name")
			result[MessageAttributes.COMPLETION_TOOL_CALL_ARGUMENTS.format(i=i)] = function_call.get("arguments")
	return result


