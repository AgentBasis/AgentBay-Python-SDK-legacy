from typing import Optional, Tuple, Dict, Any
from agentbay.bay_frameworks.instrumentation.common.attributes import AttributeMap


def get_team_run_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> AttributeMap:
	attributes: AttributeMap = {}
	try:
		if args and len(args) > 0:
			team = args[0]
			team_name = getattr(team, "name", team.__class__.__name__)
			attributes["team.name"] = team_name
			if hasattr(team, "team_id") and team.team_id:
				attributes["team.id"] = str(team.team_id)
		if return_value is not None:
			attributes["team.result"] = str(return_value)[:1000]
	except Exception:
		pass
	return attributes


