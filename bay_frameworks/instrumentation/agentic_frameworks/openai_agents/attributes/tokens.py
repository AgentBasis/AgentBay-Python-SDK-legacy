import json
from typing import Any, Dict, Optional

from agentbay.bay_frameworks.semconv import SpanAttributes
from agentbay.bay_frameworks.logging import logger


def safe_parse(content: str) -> Optional[Dict[str, Any]]:
	if not isinstance(content, str):
		return None
	try:
		return json.loads(content)
	except (json.JSONDecodeError, TypeError, ValueError):
		logger.debug(f"Failed to parse JSON content: {content[:100]}...")
		return None


def extract_nested_usage(content: Any) -> Optional[Dict[str, Any]]:
	if isinstance(content, dict) and "usage" in content:
		return content["usage"]
	if isinstance(content, str):
		parsed_data = safe_parse(content)
		if parsed_data:
			if "usage" in parsed_data and isinstance(parsed_data["usage"], dict):
				return parsed_data["usage"]
			if "output" in parsed_data and isinstance(parsed_data["output"], list):
				if "usage" in parsed_data:
					return parsed_data["usage"]
	if isinstance(content, dict):
		if "output" in content and isinstance(content["output"], list):
			if "usage" in content:
				return content["usage"]
	return None


def process_token_usage(
	usage: Dict[str, Any], attributes: Dict[str, Any], completion_content: Optional[str] = None
) -> Dict[str, Any]:
	result = {}
	if not usage or (isinstance(usage, dict) and len(usage) == 0):
		if completion_content:
			logger.debug("TOKENS: Usage is empty, trying to extract from completion content")
			extracted_usage = extract_nested_usage(completion_content)
			if extracted_usage:
				usage = extracted_usage

	def get_value(obj, key):
		if isinstance(obj, dict) and key in obj:
			return obj[key]
		elif hasattr(obj, key):
			return getattr(obj, key)
		return None

	def has_key(obj, key):
		if isinstance(obj, dict):
			return key in obj
		return hasattr(obj, key)

	if has_key(usage, "prompt_tokens"):
		prompt_tokens = get_value(usage, "prompt_tokens")
		attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
		result["prompt_tokens"] = prompt_tokens
	elif has_key(usage, "input_tokens"):
		input_tokens = get_value(usage, "input_tokens")
		attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
		result["prompt_tokens"] = input_tokens

	if has_key(usage, "completion_tokens"):
		completion_tokens = get_value(usage, "completion_tokens")
		attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
		result["completion_tokens"] = completion_tokens
	elif has_key(usage, "output_tokens"):
		output_tokens = get_value(usage, "output_tokens")
		attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
		result["completion_tokens"] = output_tokens

	if has_key(usage, "total_tokens"):
		total_tokens = get_value(usage, "total_tokens")
		attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens
		result["total_tokens"] = total_tokens

	output_tokens_details = None
	if has_key(usage, "output_tokens_details"):
		output_tokens_details = get_value(usage, "output_tokens_details")
	if output_tokens_details:
		if isinstance(output_tokens_details, dict):
			details = output_tokens_details
			if "reasoning_tokens" in details:
				attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] = details["reasoning_tokens"]
				result["reasoning_tokens"] = details["reasoning_tokens"]
		elif hasattr(output_tokens_details, "reasoning_tokens"):
			reasoning_tokens = output_tokens_details.reasoning_tokens
			attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] = reasoning_tokens
			result["reasoning_tokens"] = reasoning_tokens

	input_tokens_details = None
	if has_key(usage, "input_tokens_details"):
		input_tokens_details = get_value(usage, "input_tokens_details")
	if input_tokens_details:
		if isinstance(input_tokens_details, dict):
			details = input_tokens_details
			if "cached_tokens" in details:
				attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = details["cached_tokens"]
				result["cached_input_tokens"] = details["cached_tokens"]
		elif hasattr(input_tokens_details, "cached_tokens"):
			cached_tokens = input_tokens_details.cached_tokens
			attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cached_tokens
			result["cached_input_tokens"] = cached_tokens

	return result


