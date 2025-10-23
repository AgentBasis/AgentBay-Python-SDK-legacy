"""Core attributes applicable to all spans."""


class CoreAttributes:
	"""Core attributes applicable to all spans."""

	# Error attributes
	ERROR_TYPE = "error.type"
	ERROR_MESSAGE = "error.message"

	TAGS = "bay_frameworks.tags"

	# Trace context attributes
	TRACE_ID = "trace.id"
	SPAN_ID = "span.id"
	PARENT_ID = "parent.id"
	GROUP_ID = "group.id"

	# Note: WORKFLOW_NAME is defined in WorkflowAttributes to avoid duplication


