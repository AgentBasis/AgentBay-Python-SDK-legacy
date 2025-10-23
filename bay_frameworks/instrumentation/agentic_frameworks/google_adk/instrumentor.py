from typing import Dict, Any

from MYSDK.bay_frameworks.logging import logger
from opentelemetry.metrics import Meter
from MYSDK.bay_frameworks.instrumentation.common import CommonInstrumentor, StandardMetrics, InstrumentorConfig
from MYSDK.bay_frameworks.instrumentation.agentic_frameworks.google_adk.patch import patch_adk, unpatch_adk


LIBRARY_NAME = "bay_frameworks.instrumentation.google_adk"
LIBRARY_VERSION = "0.1.0"


class GooogleAdkInstrumentor(CommonInstrumentor):
	"""An instrumentor for Google Agent Development Kit (ADK)."""

	def __init__(self):
		config = InstrumentorConfig(
			library_name=LIBRARY_NAME,
			library_version=LIBRARY_VERSION,
			wrapped_methods=[],
			metrics_enabled=True,
			dependencies=["google-adk >= 0.1.0"],
		)
		super().__init__(config)

	def _create_metrics(self, meter: Meter) -> Dict[str, Any]:
		return StandardMetrics.create_standard_metrics(meter)

	def _custom_wrap(self, **kwargs):
		patch_adk(self._tracer)
		logger.info("Google ADK instrumentation enabled")

	def _custom_unwrap(self, **kwargs):
		unpatch_adk()
		logger.info("Google ADK instrumentation disabled")


