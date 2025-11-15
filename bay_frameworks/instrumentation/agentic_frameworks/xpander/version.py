"""Version information for xpander instrumentation (bay_frameworks)."""

# Version info (imported from main package - single source of truth)
try:
    from agentbay import __version__
except ImportError:
    # Fallback if main package not available
    __version__ = "1.0.0"


