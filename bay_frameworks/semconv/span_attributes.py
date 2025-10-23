"""Span attributes for OpenTelemetry semantic conventions."""


class SpanAttributes:
	# Gen AI conventions
	LLM_SYSTEM = "gen_ai.system"
	LLM_REQUEST_MODEL = "gen_ai.request.model"
	LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
	LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
	LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
	LLM_REQUEST_TOP_K = "gen_ai.request.top_k"
	LLM_REQUEST_SEED = "gen_ai.request.seed"
	LLM_REQUEST_SYSTEM_INSTRUCTION = "gen_ai.request.system_instruction"
	LLM_REQUEST_CANDIDATE_COUNT = "gen_ai.request.candidate_count"
	LLM_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
	LLM_REQUEST_TYPE = "gen_ai.request.type"
	LLM_REQUEST_STREAMING = "gen_ai.request.streaming"
	LLM_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
	LLM_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
	LLM_REQUEST_FUNCTIONS = "gen_ai.request.functions"
	LLM_REQUEST_HEADERS = "gen_ai.request.headers"
	LLM_REQUEST_INSTRUCTIONS = "gen_ai.request.instructions"
	LLM_REQUEST_VOICE = "gen_ai.request.voice"
	LLM_REQUEST_SPEED = "gen_ai.request.speed"

	LLM_PROMPTS = "gen_ai.prompt"
	LLM_COMPLETIONS = "gen_ai.completion"
	LLM_CONTENT_COMPLETION_CHUNK = "gen_ai.completion.chunk"

	LLM_RESPONSE_MODEL = "gen_ai.response.model"
	LLM_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"
	LLM_RESPONSE_STOP_REASON = "gen_ai.response.stop_reason"
	LLM_RESPONSE_ID = "gen_ai.response.id"

	LLM_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
	LLM_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
	LLM_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
	LLM_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation_input_tokens"
	LLM_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
	LLM_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
	LLM_USAGE_STREAMING_TOKENS = "gen_ai.usage.streaming_tokens"
	LLM_USAGE_TOOL_COST = "gen_ai.usage.total_cost"

	# OpenAI specific
	LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "gen_ai.openai.system_fingerprint"
	LLM_OPENAI_RESPONSE_INSTRUCTIONS = "gen_ai.openai.instructions"
	LLM_OPENAI_API_BASE = "gen_ai.openai.api_base"
	LLM_OPENAI_API_VERSION = "gen_ai.openai.api_version"
	LLM_OPENAI_API_TYPE = "gen_ai.openai.api_type"

	# Bay Frameworks specific
	BAYFW_ENTITY_OUTPUT = "bay_frameworks.entity.output"
	BAYFW_ENTITY_INPUT = "bay_frameworks.entity.input"
	BAYFW_SPAN_KIND = "bay_frameworks.span.kind"
	BAYFW_ENTITY_NAME = "bay_frameworks.entity.name"
	BAYFW_DECORATOR_SPEC = "bay_frameworks.{entity_kind}.spec"
	BAYFW_DECORATOR_INPUT = "bay_frameworks.{entity_kind}.input"
	BAYFW_DECORATOR_OUTPUT = "bay_frameworks.{entity_kind}.output"

	OPERATION_NAME = "operation.name"
	OPERATION_VERSION = "operation.version"

	BAYFW_SESSION_END_STATE = "bay_frameworks.session.end_state"

	# Streaming
	LLM_STREAMING_TIME_TO_FIRST_TOKEN = "gen_ai.streaming.time_to_first_token"
	LLM_STREAMING_TIME_TO_GENERATE = "gen_ai.streaming.time_to_generate"
	LLM_STREAMING_DURATION = "gen_ai.streaming_duration"
	LLM_STREAMING_CHUNK_COUNT = "gen_ai.streaming.chunk_count"

	# HTTP
	HTTP_METHOD = "http.method"
	HTTP_URL = "http.url"
	HTTP_ROUTE = "http.route"
	HTTP_STATUS_CODE = "http.status_code"
	HTTP_REQUEST_HEADERS = "http.request.headers"
	HTTP_RESPONSE_HEADERS = "http.response.headers"
	HTTP_REQUEST_BODY = "http.request.body"
	HTTP_RESPONSE_BODY = "http.response.body"
	HTTP_USER_AGENT = "http.user_agent"
	HTTP_REQUEST_ID = "http.request_id"

