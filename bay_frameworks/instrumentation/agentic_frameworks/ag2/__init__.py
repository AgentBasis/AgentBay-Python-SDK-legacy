from agentbay.bay_frameworks.instrumentation.common import LibraryInfo

_library_info = LibraryInfo(name="ag2")
LIBRARY_NAME = _library_info.name
LIBRARY_VERSION = _library_info.version

from agentbay.bay_frameworks.instrumentation.agentic_frameworks.ag2.instrumentor import AG2Instrumentor  # noqa: E402

__all__ = ["AG2Instrumentor", "LIBRARY_NAME", "LIBRARY_VERSION"]


