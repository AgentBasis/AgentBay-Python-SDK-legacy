"""Attributes specific to workflow spans."""


class WorkflowAttributes:
	"""Workflow specific attributes."""

	WORKFLOW_NAME = "workflow.name"
	WORKFLOW_TYPE = "workflow.type"
	WORKFLOW_ID = "workflow.workflow_id"
	WORKFLOW_RUN_ID = "workflow.run_id"
	WORKFLOW_DESCRIPTION = "workflow.description"

	WORKFLOW_INPUT = "workflow.input"
	WORKFLOW_INPUT_TYPE = "workflow.input.type"
	WORKFLOW_OUTPUT = "workflow.output"
	WORKFLOW_OUTPUT_TYPE = "workflow.output.type"
	WORKFLOW_FINAL_OUTPUT = "workflow.final_output"

	WORKFLOW_STEP_TYPE = "workflow.step.type"
	WORKFLOW_STEP_STATUS = "workflow.step.status"

	WORKFLOW_MAX_TURNS = "workflow.max_turns"
	WORKFLOW_DEBUG_MODE = "workflow.debug_mode"
	WORKFLOW_MONITORING = "workflow.monitoring"
	WORKFLOW_TELEMETRY = "workflow.telemetry"

	WORKFLOW_MEMORY_TYPE = "workflow.memory_type"
	WORKFLOW_STORAGE_TYPE = "workflow.storage_type"

	WORKFLOW_SESSION_ID = "workflow.session_id"
	WORKFLOW_SESSION_NAME = "workflow.session_name"
	WORKFLOW_USER_ID = "workflow.user_id"
	WORKFLOW_APP_ID = "workflow.app_id"

	WORKFLOW_INPUT_PARAMETER_COUNT = "workflow.input.parameter_count"
	WORKFLOW_INPUT_PARAMETER_KEYS = "workflow.input.parameter_keys"
	WORKFLOW_METHOD_PARAMETER_COUNT = "workflow.method.parameter_count"
	WORKFLOW_METHOD_RETURN_TYPE = "workflow.method.return_type"

	WORKFLOW_OUTPUT_CONTENT_TYPE = "workflow.output.content_type"
	WORKFLOW_OUTPUT_EVENT = "workflow.output.event"
	WORKFLOW_OUTPUT_MODEL = "workflow.output.model"
	WORKFLOW_OUTPUT_MODEL_PROVIDER = "workflow.output.model_provider"
	WORKFLOW_OUTPUT_MESSAGE_COUNT = "workflow.output.message_count"
	WORKFLOW_OUTPUT_TOOL_COUNT = "workflow.output.tool_count"
	WORKFLOW_OUTPUT_IS_STREAMING = "workflow.output.is_streaming"

	WORKFLOW_OUTPUT_IMAGE_COUNT = "workflow.output.image_count"
	WORKFLOW_OUTPUT_VIDEO_COUNT = "workflow.output.video_count"
	WORKFLOW_OUTPUT_AUDIO_COUNT = "workflow.output.audio_count"

	WORKFLOW_SESSION_WORKFLOW_ID = "workflow.session.workflow_id"
	WORKFLOW_SESSION_USER_ID = "workflow.session.user_id"
	WORKFLOW_SESSION_STATE_KEYS = "workflow.session.state_keys"
	WORKFLOW_SESSION_STATE_SIZE = "workflow.session.state_size"
	WORKFLOW_SESSION_STORAGE_TYPE = "workflow.session.storage_type"
	WORKFLOW_SESSION_RETURNED_SESSION_ID = "workflow.session.returned_session_id"
	WORKFLOW_SESSION_CREATED_AT = "workflow.session.created_at"
	WORKFLOW_SESSION_UPDATED_AT = "workflow.session.updated_at"


