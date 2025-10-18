from typing import runtime_checkable, Protocol, Any, Optional, Dict, TypedDict
from MYSDK.bay_frameworks.logging import logger
from MYSDK.bay_frameworks.helpers import safe_serialize, get_bay_frameworks_version
from MYSDK.bay_frameworks.semconv import (
	CoreAttributes,
	InstrumentationAttributes,
	WorkflowAttributes,
)


AttributeMap = Dict[str, str]


@runtime_checkable
class IndexedAttribute(Protocol):
	def format(self, *, i: int, j: Optional[int] = None) -> str: ...


IndexedAttributeMap = Dict[IndexedAttribute, str]


class IndexedAttributeData(TypedDict, total=False):
	i: int
	j: Optional[int]


def _extract_attributes_from_mapping(span_data: Any, attribute_mapping: AttributeMap) -> AttributeMap:
	attributes = {}
	for target_attr, source_attr in attribute_mapping.items():
		if hasattr(span_data, source_attr):
			value = getattr(span_data, source_attr)
		elif isinstance(span_data, dict) and source_attr in span_data:
			value = span_data[source_attr]
		else:
			continue

		if value is None or (isinstance(value, (list, dict, str)) and not value):
			continue
		elif isinstance(value, (dict, list, object)) and not isinstance(value, (str, int, float, bool)):
			value = safe_serialize(value)

		attributes[target_attr] = value

	return attributes


def _extract_attributes_from_mapping_with_index(
	span_data: Any, attribute_mapping: IndexedAttributeMap, i: int, j: Optional[int] = None
) -> AttributeMap:
	format_kwargs: IndexedAttributeData = {"i": i}
	if j is not None:
		format_kwargs["j"] = j

	attribute_mapping_with_index: AttributeMap = {}
	for target_attr, source_attr in attribute_mapping.items():
		attribute_mapping_with_index[target_attr.format(**format_kwargs)] = source_attr

	return _extract_attributes_from_mapping(span_data, attribute_mapping_with_index)


def get_common_attributes() -> AttributeMap:
	return {
		InstrumentationAttributes.NAME: "bay_frameworks",
		InstrumentationAttributes.VERSION: get_bay_frameworks_version(),
	}


def get_base_trace_attributes(trace: Any) -> AttributeMap:
	if not hasattr(trace, "trace_id"):
		logger.warning("Cannot create trace attributes: missing trace_id")
		return {}

	attributes = {
		WorkflowAttributes.WORKFLOW_NAME: trace.name,
		CoreAttributes.TRACE_ID: trace.trace_id,
		WorkflowAttributes.WORKFLOW_STEP_TYPE: "trace",
		**get_common_attributes(),
	}
	return attributes


def get_base_span_attributes(span: Any) -> AttributeMap:
	span_id = getattr(span, "span_id", "unknown")
	trace_id = getattr(span, "trace_id", "unknown")
	parent_id = getattr(span, "parent_id", None)

	attributes = {
		CoreAttributes.TRACE_ID: trace_id,
		CoreAttributes.SPAN_ID: span_id,
		**get_common_attributes(),
	}

	if parent_id:
		attributes[CoreAttributes.PARENT_ID] = parent_id

	return attributes

