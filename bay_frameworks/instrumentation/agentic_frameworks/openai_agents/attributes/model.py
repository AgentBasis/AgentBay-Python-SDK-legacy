from typing import Any, Dict
from agentbay.bay_frameworks.semconv import SpanAttributes
from agentbay.bay_frameworks.instrumentation.common.attributes import AttributeMap, _extract_attributes_from_mapping


MODEL_CONFIG_ATTRIBUTES: AttributeMap = {
	SpanAttributes.LLM_REQUEST_TEMPERATURE: "temperature",
	SpanAttributes.LLM_REQUEST_TOP_P: "top_p",
	SpanAttributes.LLM_REQUEST_FREQUENCY_PENALTY: "frequency_penalty",
	SpanAttributes.LLM_REQUEST_PRESENCE_PENALTY: "presence_penalty",
	SpanAttributes.LLM_REQUEST_MAX_TOKENS: "max_tokens",
	SpanAttributes.LLM_REQUEST_INSTRUCTIONS: "instructions",
	SpanAttributes.LLM_REQUEST_VOICE: "voice",
	SpanAttributes.LLM_REQUEST_SPEED: "speed",
}


def get_model_attributes(model_name: str) -> Dict[str, Any]:
	return {
		SpanAttributes.LLM_REQUEST_MODEL: model_name,
		SpanAttributes.LLM_RESPONSE_MODEL: model_name,
		SpanAttributes.LLM_SYSTEM: "openai",
	}


def get_model_config_attributes(model_config: Any) -> Dict[str, Any]:
	return _extract_attributes_from_mapping(model_config, MODEL_CONFIG_ATTRIBUTES)


