"""Xpander trace probe for bay_frameworks.

Helper to activate/deactivate Xpander instrumentation explicitly, though
auto-activation is handled by bay_frameworks.instrumentation orchestrator.
"""

from agentbay.bay_frameworks.instrumentation.agentic_frameworks.xpander.instrumentor import XpanderInstrumentor
from agentbay.bay_frameworks.logging import logger

_instrumentor = None


def activate_xpander_instrumentation():
	global _instrumentor
	if _instrumentor is None:
		try:
			_instrumentor = XpanderInstrumentor()
			_instrumentor.instrument()
			logger.info("Xpander instrumentation activated successfully")
		except Exception as e:
			logger.error(f"Failed to activate Xpander instrumentation: {e}")
			_instrumentor = None
	return _instrumentor


def deactivate_xpander_instrumentation():
	global _instrumentor
	if _instrumentor is not None:
		try:
			_instrumentor.uninstrument()
			logger.info("Xpander instrumentation deactivated successfully")
		except Exception as e:
			logger.error(f"Failed to deactivate Xpander instrumentation: {e}")
		finally:
			_instrumentor = None


def get_instrumentor():
	return _instrumentor


def wrap_openai_call_for_xpander(openai_call_func, purpose="general"):
	logger.debug(f"wrap_openai_call_for_xpander called with purpose: {purpose}")
	return openai_call_func


def is_xpander_session_active():
	return _instrumentor is not None


def get_active_xpander_session():
	return _instrumentor._context if _instrumentor else None


def wrap_openai_analysis(openai_call_func):
	return wrap_openai_call_for_xpander(openai_call_func, "analysis")


def wrap_openai_planning(openai_call_func):
	return wrap_openai_call_for_xpander(openai_call_func, "planning")


def wrap_openai_synthesis(openai_call_func):
	return wrap_openai_call_for_xpander(openai_call_func, "synthesis")


