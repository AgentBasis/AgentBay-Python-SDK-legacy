"""Bay Frameworks instrumentation namespace.

This package provides monitoring-only instrumentation for agentic frameworks.
Import `agentbay.bay_frameworks.instrumentation` and call `instrument_all()` to
enable auto-instrumentation of supported agentic frameworks.
"""

# Public re-exports (lazy users may import this module root)
try:
	from . import instrumentation as instrumentation  # noqa: F401
except Exception:
	# If submodules are not yet available, importing root should not fail.
	pass

