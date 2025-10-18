from typing import Optional, Tuple, Dict, Any
from MYSDK.bay_frameworks.instrumentation.common.attributes import AttributeMap
from MYSDK.bay_frameworks.semconv import ToolAttributes
import json


def get_tool_execution_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			instance = args[0]
			if hasattr(instance, "name"):
				attributes[ToolAttributes.TOOL_NAME] = instance.name
			if hasattr(instance, "description"):
				attributes[ToolAttributes.TOOL_DESCRIPTION] = instance.description
		if kwargs and "tool_call" in kwargs:
			tool_call = kwargs["tool_call"]
			if hasattr(tool_call, "function"):
				attributes[ToolAttributes.TOOL_NAME] = tool_call.function.name
				if hasattr(tool_call.function, "arguments"):
					attributes[ToolAttributes.TOOL_PARAMETERS] = tool_call.function.arguments
		if kwargs:
			for k in ("parameters", "arguments", "args"):
				if k in kwargs:
					try:
						attributes[ToolAttributes.TOOL_PARAMETERS] = json.dumps(kwargs[k])
					except Exception:
						attributes[ToolAttributes.TOOL_PARAMETERS] = str(kwargs[k])
		if return_value is not None:
			attributes[ToolAttributes.TOOL_RESULT] = str(return_value)
			attributes[ToolAttributes.TOOL_STATUS] = "success"
	except Exception:
		pass
	return attributes


