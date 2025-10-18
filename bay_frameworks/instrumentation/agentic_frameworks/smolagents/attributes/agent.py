from typing import Any, Dict, Optional, Tuple
import uuid
import json

from MYSDK.bay_frameworks.instrumentation.common.attributes import get_common_attributes
from MYSDK.bay_frameworks.semconv.agent import AgentAttributes
from MYSDK.bay_frameworks.semconv.tool import ToolAttributes


def get_agent_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		agent_instance = None
		if args and len(args) > 0:
			agent_instance = args[0]
		elif kwargs and "self" in kwargs:
			agent_instance = kwargs["self"]
		if agent_instance:
			agent_name = getattr(agent_instance, "name", agent_instance.__class__.__name__)
			attributes[AgentAttributes.AGENT_NAME] = agent_name
			agent_id = getattr(agent_instance, "id", str(uuid.uuid4()))
			attributes[AgentAttributes.AGENT_ID] = agent_id
			attributes[AgentAttributes.AGENT_ROLE] = "executor"
			tools = getattr(agent_instance, "tools", [])
			if tools:
				tool_names = []
				for tool in tools:
					tool_name = getattr(tool, "name", str(tool))
					tool_names.append(tool_name)
				attributes[AgentAttributes.AGENT_TOOLS] = json.dumps(tool_names)
			else:
				attributes[AgentAttributes.AGENT_TOOLS] = "[]"
		task_input = None
		if args and len(args) > 1:
			task_input = args[1]
		elif kwargs and "task" in kwargs:
			task_input = kwargs["task"]
		elif kwargs and "prompt" in kwargs:
			task_input = kwargs["prompt"]
		if task_input:
			attributes["agent.task"] = str(task_input)
		if return_value is not None:
			attributes["bay_frameworks.entity.output"] = str(return_value)
	except Exception:
		pass
	return attributes


def get_agent_stream_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		if kwargs:
			if "max_steps" in kwargs:
				attributes["agent.max_steps"] = str(kwargs["max_steps"])
			for param_name in ["task", "prompt", "reasoning", "query"]:
				if param_name in kwargs:
					attributes["agent.reasoning"] = str(kwargs[param_name])
		if args and len(args) > 1:
			attributes["agent.reasoning"] = str(args[1])
	except Exception:
		pass
	return attributes


def get_agent_step_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		step_number = getattr(args[0] if args else None, "step_count", None)
		if step_number is not None:
			attributes["agent.step_number"] = str(step_number)
		attributes["agent.name"] = "ActionStep"
		if return_value is not None:
			attributes["bay_frameworks.entity.output"] = str(return_value)
	except Exception:
		pass
	return attributes


def get_tool_call_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		tool_id = str(uuid.uuid4())
		attributes[ToolAttributes.TOOL_ID] = tool_id
		tool_name = "unknown"
		tool_description = "unknown"
		tool_parameters = "{}"
		if args and len(args) > 0:
			instance = args[0]
			if hasattr(instance, "name"):
				tool_name = instance.name
			if hasattr(instance, "description"):
				tool_description = instance.description
		if kwargs:
			if "tool_call" in kwargs and hasattr(kwargs["tool_call"], "function"):
				tool_name = kwargs["tool_call"].function.name
				if hasattr(kwargs["tool_call"].function, "arguments"):
					tool_parameters = kwargs["tool_call"].function.arguments
			elif "name" in kwargs:
				tool_name = kwargs["name"]
			elif "function_name" in kwargs:
				tool_name = kwargs["function_name"]
			if "parameters" in kwargs:
				tool_parameters = json.dumps(kwargs["parameters"])
			elif "arguments" in kwargs:
				tool_parameters = json.dumps(kwargs["arguments"])
			elif "args" in kwargs:
				tool_parameters = json.dumps(kwargs["args"])
		attributes[ToolAttributes.TOOL_NAME] = tool_name
		attributes[ToolAttributes.TOOL_DESCRIPTION] = tool_description
		attributes[ToolAttributes.TOOL_PARAMETERS] = tool_parameters
		attributes[ToolAttributes.TOOL_STATUS] = "pending"
		attributes[ToolAttributes.TOOL_OUTPUT_TYPE] = "unknown"
		attributes[ToolAttributes.TOOL_INPUTS] = "{}"
		if return_value is not None:
			attributes["tool.result"] = str(return_value)
			attributes[ToolAttributes.TOOL_STATUS] = "success"
	except Exception:
		attributes[ToolAttributes.TOOL_NAME] = "unknown"
		attributes[ToolAttributes.TOOL_DESCRIPTION] = "unknown"
		attributes[ToolAttributes.TOOL_ID] = str(uuid.uuid4())
		attributes[ToolAttributes.TOOL_PARAMETERS] = "{}"
		attributes[ToolAttributes.TOOL_STATUS] = "pending"
	return attributes


def get_planning_step_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		if kwargs and "planning_step" in kwargs:
			attributes["agent.planning.step"] = str(kwargs["planning_step"])
		if kwargs and "reasoning" in kwargs:
			attributes["agent.planning.reasoning"] = str(kwargs["reasoning"])
		if return_value is not None:
			attributes["bay_frameworks.entity.output"] = str(return_value)
	except Exception:
		pass
	return attributes


def get_managed_agent_attributes(
	args: Optional[Tuple] = None,
	kwargs: Optional[Dict] = None,
	return_value: Optional[Any] = None,
) -> Dict[str, Any]:
	attributes = get_common_attributes()
	try:
		agent_instance = None
		if args and len(args) > 0:
			agent_instance = args[0]
		elif kwargs and "agent" in kwargs:
			agent_instance = kwargs["agent"]
		if agent_instance:
			agent_name = getattr(agent_instance, "name", agent_instance.__class__.__name__)
			agent_id = getattr(agent_instance, "id", str(uuid.uuid4()))
			agent_description = getattr(agent_instance, "description", "")
			attributes[AgentAttributes.AGENT_NAME] = agent_name
			attributes[AgentAttributes.AGENT_ID] = agent_id
			attributes[AgentAttributes.AGENT_ROLE] = "managed"
			attributes[AgentAttributes.AGENT_TYPE] = agent_instance.__class__.__name__
			if agent_description:
				attributes["agent.description"] = agent_description
			attributes["agent.provide_run_summary"] = "false"
		if args and len(args) > 1:
			attributes["agent.task"] = str(args[1])
		elif kwargs and "task" in kwargs:
			attributes["agent.task"] = str(kwargs["task"])
		if return_value is not None:
			attributes["bay_frameworks.entity.output"] = str(return_value)
	except Exception:
		pass
	return attributes


