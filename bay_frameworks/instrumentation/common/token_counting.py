from dataclasses import dataclass
from typing import Optional, Any
from agentbay.bay_frameworks.semconv import SpanAttributes


@dataclass
class TokenUsage:
	prompt_tokens: Optional[int] = None
	completion_tokens: Optional[int] = None
	total_tokens: Optional[int] = None
	cached_prompt_tokens: Optional[int] = None


class TokenUsageExtractor:
	@staticmethod
	def extract_from_response(response: Any) -> TokenUsage:
		usage = TokenUsage()
		if hasattr(response, "usage") and response.usage:
			u = response.usage
			usage.prompt_tokens = getattr(u, "prompt_tokens", None)
			usage.completion_tokens = getattr(u, "completion_tokens", None)
			usage.total_tokens = getattr(u, "total_tokens", None)
			if hasattr(u, "input_tokens_details") and getattr(u, "input_tokens_details", None):
				usage.cached_prompt_tokens = getattr(u.input_tokens_details, "cached_tokens", None)
		return usage


def calculate_token_efficiency(usage: TokenUsage) -> Optional[float]:
	if usage.prompt_tokens and usage.completion_tokens and usage.prompt_tokens > 0:
		return usage.completion_tokens / usage.prompt_tokens
	return None


def calculate_cache_efficiency(usage: TokenUsage) -> Optional[float]:
	if usage.cached_prompt_tokens and usage.prompt_tokens and usage.prompt_tokens > 0:
		return usage.cached_prompt_tokens / usage.prompt_tokens
	return None


def set_token_usage_attributes(span, result):
	if not hasattr(result, "usage") or not result.usage:
		return
	u = result.usage
	if hasattr(u, "completion_tokens") and u.completion_tokens is not None:
		span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, u.completion_tokens)
	if hasattr(u, "prompt_tokens") and u.prompt_tokens is not None:
		span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, u.prompt_tokens)
	if hasattr(u, "total_tokens") and u.total_tokens is not None:
		span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, u.total_tokens)
	if hasattr(u, "input_tokens_details") and getattr(u, "input_tokens_details", None):
		cached = getattr(u.input_tokens_details, "cached_tokens", None)
		if cached is not None:
			span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)


