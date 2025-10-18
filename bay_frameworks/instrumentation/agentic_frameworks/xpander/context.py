"""Xpander context management for session tracking (bay_frameworks)."""

import time
import threading
from typing import Any, Dict, Optional


class XpanderContext:
	"""Context manager for Xpander sessions with nested conversation spans."""

	def __init__(self):
		self._sessions = {}
		self._workflow_spans = {}
		self._agent_spans = {}
		self._conversation_spans = {}
		self._conversation_counters = {}
		self._lock = threading.Lock()

	def start_session(self, session_id: str, agent_info: Dict[str, Any], workflow_span=None, agent_span=None) -> None:
		with self._lock:
			self._sessions[session_id] = {
				"agent_name": agent_info.get("agent_name", "unknown"),
				"agent_id": agent_info.get("agent_id", "unknown"),
				"task_input": agent_info.get("task_input"),
				"phase": "planning",
				"step_count": 0,
				"total_tokens": 0,
				"tools_executed": [],
				"start_time": time.time(),
			}
			if workflow_span:
				self._workflow_spans[session_id] = workflow_span
			if agent_span:
				self._agent_spans[session_id] = agent_span
			self._conversation_counters[session_id] = 0

	def start_conversation(self, session_id: str, conversation_span) -> None:
		with self._lock:
			self._conversation_spans[session_id] = conversation_span
			self._conversation_counters[session_id] = self._conversation_counters.get(session_id, 0) + 1

	def end_conversation(self, session_id: str) -> None:
		with self._lock:
			if session_id in self._conversation_spans:
				del self._conversation_spans[session_id]

	def has_active_conversation(self, session_id: str) -> bool:
		with self._lock:
			return session_id in self._conversation_spans

	def get_conversation_counter(self, session_id: str) -> int:
		with self._lock:
			return self._conversation_counters.get(session_id, 0)

	def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
		with self._lock:
			return self._sessions.get(session_id)

	def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
		with self._lock:
			if session_id in self._sessions:
				self._sessions[session_id].update(updates)

	def end_session(self, session_id: str) -> None:
		with self._lock:
			self._sessions.pop(session_id, None)
			self._workflow_spans.pop(session_id, None)
			self._agent_spans.pop(session_id, None)
			self._conversation_spans.pop(session_id, None)
			self._conversation_counters.pop(session_id, None)

	def get_workflow_phase(self, session_id: str) -> str:
		with self._lock:
			session = self._sessions.get(session_id, {})
			if session.get("tools_executed", []):
				return "executing"
			elif session.get("step_count", 0) > 0:
				return "executing"
			return "planning"

	def get_workflow_span(self, session_id: str):
		with self._lock:
			return self._workflow_spans.get(session_id)

	def get_agent_span(self, session_id: str):
		with self._lock:
			return self._agent_spans.get(session_id)

	def get_conversation_span(self, session_id: str):
		with self._lock:
			return self._conversation_spans.get(session_id)


