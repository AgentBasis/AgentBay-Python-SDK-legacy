"""Serialization helpers for bay_frameworks."""

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from MYSDK.bay_frameworks.logging import logger


def is_jsonable(x):
	try:
		json.dumps(x)
		return True
	except (TypeError, OverflowError):
		return False


def model_to_dict(obj: Any) -> dict:
	if obj is None:
		return {}
	if isinstance(obj, dict):
		return obj
	if hasattr(obj, "model_dump"):
		return obj.model_dump()
	elif hasattr(obj, "dict"):
		return obj.dict()
	try:
		return obj.__dict__
	except Exception:
		return {}


class BayFrameworksJSONEncoder(json.JSONEncoder):
	def default(self, obj: Any) -> Any:
		if isinstance(obj, UUID):
			return str(obj)
		if isinstance(obj, datetime):
			return obj.isoformat()
		if isinstance(obj, Decimal):
			return str(obj)
		if isinstance(obj, set):
			return list(obj)
		if hasattr(obj, "to_json"):
			return obj.to_json()
		if isinstance(obj, Enum):
			return obj.value
		return str(obj)


def safe_serialize(obj: Any) -> Any:
	"""Safely serialize an object to JSON-compatible format.

	- Strings passed through unchanged.
	- Models coerced to dictionaries when possible.
	- Custom encoder handles common non-JSON primitives.
	"""
	if isinstance(obj, str):
		return obj
	if hasattr(obj, "model_dump") or hasattr(obj, "dict"):
		obj = model_to_dict(obj)
	try:
		return json.dumps(obj, cls=BayFrameworksJSONEncoder)
	except (TypeError, ValueError) as e:
		logger.warning(f"Failed to serialize object: {e}")
		return str(obj)

