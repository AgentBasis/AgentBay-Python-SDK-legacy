"""Agentic-only import-hook orchestrator for bay_frameworks.

- Only auto-instruments agentic frameworks (no providers)
- Guards against double-wrapping via registry and in-flight set
- Skips agentbay.Tracker_llm and known provider namespaces to avoid collisions
"""

from typing import Optional, Set, TypedDict
from dataclasses import dataclass
import builtins
import importlib
import sys

try:
	from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
except Exception:  # pragma: no cover
	# Allow importing without OTel installed; users should install deps when using instrumentation
	class BaseInstrumentor:  # type: ignore
		def instrument(self, **kwargs):
			pass

		def uninstrument(self, **kwargs):
			pass


class InstrumentorConfig(TypedDict):
	module_name: str
	class_name: str
	min_version: str
	package_name: Optional[str]


# Supported agentic frameworks mapping (module paths to load instrumentors from)
AGENTIC_LIBRARIES: dict[str, InstrumentorConfig] = {
	"crewai": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.crewai",
		"class_name": "CrewaiInstrumentor",
		"min_version": "0.56.0",
		"package_name": None,
	},
	"ag2": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.ag2",
		"class_name": "AG2Instrumentor",
		"min_version": "0.3.2",
		"package_name": None,
	},
	"agents": {  # OpenAI Agents
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.openai_agents",
		"class_name": "OpenAIAgentsInstrumentor",
		"min_version": "0.0.1",
		"package_name": None,
	},
	"google.adk": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.google_adk",
		"class_name": "GooogleAdkInstrumentor",
		"min_version": "0.1.0",
		"package_name": None,
	},
	"agno": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.agno",
		"class_name": "AgnoInstrumentor",
		"min_version": "1.5.8",
		"package_name": None,
	},
	"smolagents": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.smolagents",
		"class_name": "SmolagentsInstrumentor",
		"min_version": "1.0.0",
		"package_name": None,
	},
	"langgraph": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.langgraph",
		"class_name": "LanggraphInstrumentor",
		"min_version": "0.2.0",
		"package_name": None,
	},
	"xpander_sdk": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.xpander",
		"class_name": "XpanderInstrumentor",
		"min_version": "1.0.0",
		"package_name": "xpander-sdk",
	},
	"haystack": {
		"module_name": "agentbay.bay_frameworks.instrumentation.agentic_frameworks.haystack",
		"class_name": "HaystackInstrumentor",
		"min_version": "2.0.0",
		"package_name": "haystack-ai",
	},
}


# Skip list: providers and local Tracker_llm
_SKIP_MODULE_PREFIXES: Set[str] = {
	"agentbay.Tracker_llm",
	"openai.resources",
	"openai.types",
	"openai._",
	"anthropic",
	"google.genai",
	"google.generativeai",
	"ibm_watsonx_ai",
	"mem0",
	"xai",
	"groq",
	"mistralai",
	"cohere",
}


_active_instrumentors: list[BaseInstrumentor] = []
_instrumented_packages: Set[str] = set()
_instrumenting_packages: Set[str] = set()
_original_import = builtins.__import__


def _is_package_instrumented(pkg: str) -> bool:
	if pkg in _instrumented_packages:
		return True
	for inst in _active_instrumentors:
		key = getattr(inst, "_bay_frameworks_instrumented_package_key", None)
		if key == pkg:
			return True
	return False


def _should_instrument(pkg: str) -> bool:
	if pkg not in AGENTIC_LIBRARIES:
		return False
	if _is_package_instrumented(pkg):
		return False
	return True


def _perform_instrumentation(pkg: str, tracer_provider=None, meter_provider=None) -> None:
	if not _should_instrument(pkg):
		return
	cfg = AGENTIC_LIBRARIES[pkg]
	module = importlib.import_module(cfg["module_name"])  # type: ignore[arg-type]
	instrumentor: BaseInstrumentor = getattr(module, cfg["class_name"])()
	setattr(instrumentor, "_bay_frameworks_instrumented_package_key", pkg)
	try:
		instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
	except TypeError:
		# Some instrumentors only accept tracer_provider
		instrumentor.instrument(tracer_provider=tracer_provider)
	_active_instrumentors.append(instrumentor)
	_instrumented_packages.add(pkg)


def _import_monitor(name, globals=None, locals=None, fromlist=(), level=0):
	for p in _SKIP_MODULE_PREFIXES:
		if isinstance(name, str) and name.startswith(p):
			return _original_import(name, globals, locals, fromlist, level)

	module = _original_import(name, globals, locals, fromlist, level)

	# Identify candidate agentic packages
	candidates = set()
	for k in AGENTIC_LIBRARIES.keys():
		if name == k or (isinstance(name, str) and name.startswith(k + ".")):
			candidates.add(k)

	for pkg in candidates:
		if pkg in _instrumenting_packages or _is_package_instrumented(pkg):
			continue
		_instrumenting_packages.add(pkg)
		try:
			_perform_instrumentation(pkg)
		finally:
			_instrumenting_packages.discard(pkg)

	return module


def instrument_all(tracer_provider=None, meter_provider=None) -> None:
	"""
	Auto-instrument all detected agentic frameworks.
	
	Scans for supported frameworks (langgraph, crewai, ag2, agno, smolagents, etc.)
	and automatically enables instrumentation when detected.
	
	Args:
		tracer_provider: Optional OpenTelemetry tracer provider
		meter_provider: Optional OpenTelemetry meter provider
	"""
	if builtins.__import__ is not _import_monitor:
		builtins.__import__ = _import_monitor
	# Scan already-imported modules
	for name in list(sys.modules.keys()):
		if any(name.startswith(p) for p in _SKIP_MODULE_PREFIXES):
			continue
		for pkg in AGENTIC_LIBRARIES.keys():
			if name == pkg or name.startswith(pkg + "."):
				_perform_instrumentation(pkg, tracer_provider, meter_provider)


def uninstrument_all() -> None:
	"""Disable instrumentation for all agentic frameworks."""
	builtins.__import__ = _original_import
	for inst in _active_instrumentors:
		try:
			inst.uninstrument()
		except Exception:
			pass
	_active_instrumentors.clear()
	_instrumented_packages.clear()


def get_active_libraries() -> Set[str]:
	"""
	Get list of currently instrumented agentic frameworks.
	
	Returns:
		Set of framework package names that are currently instrumented
	"""
	active: Set[str] = set()
	for name, module in sys.modules.items():
		if not isinstance(name, str):
			continue
		for pkg in AGENTIC_LIBRARIES.keys():
			if name == pkg or name.startswith(pkg + "."):
				active.add(pkg)
	return active
