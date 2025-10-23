from typing import Optional, Tuple, Dict, Any
from MYSDK.bay_frameworks.instrumentation.common.attributes import AttributeMap


def get_storage_read_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			workflow = args[0]
			if hasattr(workflow, "session_state") and isinstance(workflow.session_state, dict):
				attributes["storage.cache_size"] = len(workflow.session_state)
		if kwargs and "key" in kwargs:
			attributes["storage.key"] = str(kwargs["key"])[:200]
		if return_value is not None:
			attributes["storage.result"] = str(return_value)[:500]
	except Exception:
		pass
	return attributes


def get_storage_write_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			workflow = args[0]
			if hasattr(workflow, "session_state") and isinstance(workflow.session_state, dict):
				attributes["storage.cache_size"] = len(workflow.session_state)
		if kwargs and "key" in kwargs:
			attributes["storage.key"] = str(kwargs["key"])[:200]
		if kwargs and "value" in kwargs:
			attributes["storage.value_preview"] = str(kwargs["value"])[:100]
		if return_value is not None:
			attributes["storage.result"] = str(return_value)[:500]
	except Exception:
		pass
	return attributes


