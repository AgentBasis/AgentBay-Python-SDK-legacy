from typing import Optional, Tuple, Dict, Any
from MYSDK.bay_frameworks.instrumentation.common.attributes import AttributeMap
from MYSDK.bay_frameworks.semconv import WorkflowAttributes


def get_workflow_run_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			workflow = args[0]
			name = getattr(workflow, "name", workflow.__class__.__name__)
			attributes[WorkflowAttributes.WORKFLOW_NAME] = name
			if hasattr(workflow, "workflow_id") and workflow.workflow_id:
				attributes["workflow.workflow_id"] = str(workflow.workflow_id)
		if kwargs and "input" in kwargs:
			attributes[WorkflowAttributes.WORKFLOW_INPUT] = str(kwargs["input"])[:1000]
		if return_value is not None:
			attributes[WorkflowAttributes.WORKFLOW_OUTPUT] = str(return_value)[:1000]
	except Exception:
		pass
	return attributes


def get_workflow_session_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			workflow = args[0]
			attributes["workflow.session.name"] = getattr(workflow, "name", workflow.__class__.__name__)
			if hasattr(workflow, "session_id") and workflow.session_id:
				attributes["workflow.session_id"] = str(workflow.session_id)
		if kwargs:
			for k in ("session_id", "user_id"):
				if k in kwargs:
					attributes[f"workflow.session.{k}"] = str(kwargs[k])
		if return_value is not None:
			attributes["workflow.session.result"] = str(return_value)[:1000]
	except Exception:
		pass
	return attributes


