from typing import Any, Dict
from opentelemetry.metrics import get_meter, Meter
from agentbay.bay_frameworks.semconv import Meters


class MetricsRecorder:
	pass


class StandardMetrics:
	@staticmethod
	def create_standard_metrics(meter: Meter | None = None) -> Dict[str, Any]:
		if meter is None:
			meter = get_meter("bay_frameworks", version="0.1.0")
		metrics: Dict[str, Any] = {}
		try:
			metrics["token_histogram"] = meter.create_histogram(Meters.TOKEN_HISTOGRAM)
		except Exception:
			metrics["token_histogram"] = None
		try:
			metrics["duration_histogram"] = meter.create_histogram(Meters.DURATION_HISTOGRAM)
		except Exception:
			metrics["duration_histogram"] = None
		return metrics


