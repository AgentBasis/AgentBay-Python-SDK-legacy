from typing import Optional, Tuple, Dict, Any
from agentbay.bay_frameworks.instrumentation.common.attributes import AttributeMap


def get_metrics_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if kwargs and "metrics" in kwargs:
			attributes["agno.metrics.raw"] = str(kwargs["metrics"])[:1000]
	except Exception:
		pass
	return attributes


