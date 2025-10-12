"""
Security telemetry utilities.

Provides lightweight helpers for routing detector telemetry events to
application logging, in-memory buffers for testing, or external systems.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class TelemetryEvent:
    """Container for detector telemetry data."""

    payload: Dict[str, Any]
    emitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SecurityTelemetryEmitter:
    """
    Collects telemetry emitted by detectors.

    Stores a bounded buffer of the most recent events and forwards them to an
    optional logger, enabling both programmatic inspection (e.g., tests) and
    operational observability.
    """

    def __init__(self, max_events: int = 500, logger: Optional[logging.Logger] = None):
        self.max_events = max_events
        self._logger = logger or logging.getLogger("security_monitor.telemetry")
        self._events: Deque[TelemetryEvent] = deque(maxlen=max_events)

    @property
    def events(self) -> List[TelemetryEvent]:
        """Return a snapshot of collected telemetry events."""
        return list(self._events)

    def emit(self, payload: Dict[str, Any]) -> None:
        """Capture a telemetry payload."""
        if not isinstance(payload, dict):
            self._logger.warning("Ignoring non-dict telemetry payload: %r", payload)
            return

        event = TelemetryEvent(payload=dict(payload))
        self._events.append(event)
        self._logger.debug("Detector telemetry event captured: %s", event.payload)

    def handler(self) -> Callable[[Dict[str, Any]], None]:
        """Callable suitable for passing to detector telemetry_handler."""
        return self.emit


def create_logging_telemetry_handler(logger: Optional[logging.Logger] = None) -> Callable[[Dict[str, Any]], None]:
    """
    Create a simple telemetry handler that logs detector events.

    Args:
        logger: Optional logger to use. Defaults to 'security_monitor.telemetry'.
    """
    log = logger or logging.getLogger("security_monitor.telemetry")

    def _handler(payload: Dict[str, Any]) -> None:
        log.info("Security telemetry event: %s", payload)

    return _handler
