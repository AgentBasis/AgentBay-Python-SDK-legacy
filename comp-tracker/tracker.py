"""
Main system component tracker for AgentBay.

This module provides the main instrumentation functions for system component tracking,
including CPU, memory, disk, network, and process monitoring using OpenTelemetry.
"""

import atexit
import threading
import time
from typing import Optional, Dict, Any, Callable, List
from contextlib import contextmanager

from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

from .config import logger, get_config, should_collect_metric, get_collection_interval
from .system import get_host_env, get_system_snapshot
from .attributes import (
    get_system_resource_attributes,
    get_process_resource_attributes,
    get_network_resource_attributes,
    get_global_resource_attributes,
)
from .metrics import (
    SystemMetricsRecorder,
    PeriodicMetricsCollector,
    get_metrics_recorder,
    start_periodic_collection,
    stop_periodic_collection,
    record_system_snapshot,
)
from .resource import ResourceAttributes, SystemAttributes


class SystemTracker:
    """Main system component tracker for AgentBay."""
    
    def __init__(self, service_name: str = "agentbay-system"):
        self.service_name = service_name
        self.tracer = get_tracer("agentbay.system")
        self.meter = metrics.get_meter("agentbay.system")
        self._initialized = False
        self._collectors: Dict[str, PeriodicMetricsCollector] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
    def initialize(self, **kwargs) -> None:
        """
        Initialize the system tracker.
        
        Args:
            **kwargs: Additional configuration options
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            logger.info("Initializing AgentBay system tracker")
            
            # Register cleanup on exit
            atexit.register(self.shutdown)
            
            self._initialized = True
            logger.info("AgentBay system tracker initialized successfully")
    
    def start_monitoring(self, interval: Optional[float] = None) -> None:
        """
        Start system monitoring with periodic collection.
        
        Args:
            interval: Collection interval in seconds (uses config default if None)
        """
        if not self._initialized:
            self.initialize()
            
        config = get_config()
        collection_interval = interval or config.default_collection_interval
        
        logger.info(f"Starting system monitoring (interval: {collection_interval}s)")
        
        # Start periodic metrics collection
        start_periodic_collection(collection_interval)
        
        # Start monitoring thread for spans and events
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(collection_interval,),
                daemon=True
            )
            self._monitoring_thread.start()
            
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        logger.info("Stopping system monitoring")
        
        # Stop periodic collection
        stop_periodic_collection()
        
        # Stop monitoring thread
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop that runs in background thread."""
        while not self._stop_monitoring.wait(interval):
            try:
                self._collect_system_spans()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_spans(self) -> None:
        """Collect system resource information as spans."""
        config = get_config()
        
        with self.tracer.start_as_current_span(
            "system.resource.collection",
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                # Set basic span attributes
                span.set_attribute(SystemAttributes.OPERATION_NAME, "system_monitoring")
                span.set_attribute(SystemAttributes.SPAN_KIND, "system.monitoring")
                
                # Collect system attributes
                system_attrs = get_system_resource_attributes()
                for key, value in system_attrs.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        logger.debug(f"Error setting span attribute {key}: {e}")
                
                # Collect process attributes if enabled
                if should_collect_metric('process'):
                    process_attrs = get_process_resource_attributes()
                    for key, value in process_attrs.items():
                        try:
                            span.set_attribute(key, value)
                        except Exception as e:
                            logger.debug(f"Error setting process attribute {key}: {e}")
                
                # Collect network attributes if enabled
                if should_collect_metric('network'):
                    network_attrs = get_network_resource_attributes()
                    for key, value in network_attrs.items():
                        try:
                            span.set_attribute(key, value)
                        except Exception as e:
                            logger.debug(f"Error setting network attribute {key}: {e}")
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error collecting system spans: {e}")
    
    @contextmanager
    def trace_system_operation(self, operation_name: str, **attributes):
        """
        Context manager for tracing system operations.
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional attributes to set on the span
        """
        with self.tracer.start_as_current_span(
            f"system.{operation_name}",
            kind=SpanKind.INTERNAL
        ) as span:
            try:
                # Set operation attributes
                span.set_attribute(SystemAttributes.OPERATION_NAME, operation_name)
                span.set_attribute(SystemAttributes.SPAN_KIND, "system.operation")
                
                # Set additional attributes
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        logger.debug(f"Error setting attribute {key}: {e}")
                
                yield span
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def record_system_event(self, event_name: str, **attributes) -> None:
        """
        Record a system event as a span.
        
        Args:
            event_name: Name of the event
            **attributes: Event attributes
        """
        with self.tracer.start_as_current_span(
            f"system.event.{event_name}",
            kind=SpanKind.INTERNAL
        ) as span:
            span.set_attribute(SystemAttributes.OPERATION_NAME, event_name)
            span.set_attribute(SystemAttributes.SPAN_KIND, "system.event")
            
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception as e:
                    logger.debug(f"Error setting event attribute {key}: {e}")
    
    def get_system_info(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get current system information.
        
        Args:
            detailed: Include detailed system information
            
        Returns:
            Dictionary containing system information
        """
        config = get_config()
        
        if detailed:
            return get_host_env(opt_out=config.privacy_mode)
        else:
            return get_system_snapshot()
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check system health and return status.
        
        Returns:
            Dictionary containing system health status
        """
        config = get_config()
        health_status = {"status": "healthy", "alerts": []}
        
        try:
            import psutil
            
            # Check CPU usage
            if should_collect_metric('cpu'):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent > config.cpu_alert_threshold:
                    health_status["alerts"].append({
                        "type": "cpu",
                        "severity": "warning",
                        "message": f"High CPU usage: {cpu_percent}%",
                        "threshold": config.cpu_alert_threshold,
                        "current": cpu_percent,
                    })
            
            # Check memory usage
            if should_collect_metric('memory'):
                memory = psutil.virtual_memory()
                if memory.percent > config.memory_alert_threshold:
                    health_status["alerts"].append({
                        "type": "memory",
                        "severity": "warning",
                        "message": f"High memory usage: {memory.percent}%",
                        "threshold": config.memory_alert_threshold,
                        "current": memory.percent,
                    })
            
            # Check disk usage
            if should_collect_metric('disk'):
                try:
                    disk_usage = psutil.disk_usage('/')
                    disk_percent = (disk_usage.used / disk_usage.total) * 100
                    if disk_percent > config.disk_alert_threshold:
                        health_status["alerts"].append({
                            "type": "disk",
                            "severity": "warning",
                            "message": f"High disk usage: {disk_percent:.1f}%",
                            "threshold": config.disk_alert_threshold,
                            "current": disk_percent,
                        })
                except Exception:
                    pass
            
            # Set overall status
            if health_status["alerts"]:
                health_status["status"] = "warning"
                
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Error checking system health: {e}")
        
        return health_status
    
    def shutdown(self) -> None:
        """Shutdown the system tracker and clean up resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down AgentBay system tracker")
        
        try:
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        self._initialized = False
        logger.info("AgentBay system tracker shutdown complete")


# Global tracker instance
_tracker: Optional[SystemTracker] = None
_tracker_lock = threading.Lock()


def get_system_tracker() -> SystemTracker:
    """Get the global system tracker instance."""
    global _tracker
    
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = SystemTracker()
    
    return _tracker


def instrument_system(service_name: str = "agentbay-system", **kwargs) -> None:
    """
    Instrument system component tracking.
    
    Args:
        service_name: Name of the service for telemetry
        **kwargs: Additional configuration options
    """
    tracker = get_system_tracker()
    if service_name != tracker.service_name:
        tracker.service_name = service_name
        
    tracker.initialize(**kwargs)
    logger.info("AgentBay system instrumentation enabled")


def start_system_monitoring(interval: Optional[float] = None) -> None:
    """
    Start system monitoring.
    
    Args:
        interval: Collection interval in seconds
    """
    tracker = get_system_tracker()
    tracker.start_monitoring(interval)


def stop_system_monitoring() -> None:
    """Stop system monitoring."""
    tracker = get_system_tracker()
    tracker.stop_monitoring()


def trace_system_operation(operation_name: str, **attributes):
    """
    Context manager for tracing system operations.
    
    Args:
        operation_name: Name of the operation
        **attributes: Additional attributes
    """
    tracker = get_system_tracker()
    return tracker.trace_system_operation(operation_name, **attributes)


def record_system_event(event_name: str, **attributes) -> None:
    """
    Record a system event.
    
    Args:
        event_name: Name of the event
        **attributes: Event attributes
    """
    tracker = get_system_tracker()
    tracker.record_system_event(event_name, **attributes)


def get_system_info(detailed: bool = False) -> Dict[str, Any]:
    """
    Get current system information.
    
    Args:
        detailed: Include detailed system information
        
    Returns:
        Dictionary containing system information
    """
    tracker = get_system_tracker()
    return tracker.get_system_info(detailed)


def check_system_health() -> Dict[str, Any]:
    """
    Check system health and return status.
    
    Returns:
        Dictionary containing system health status
    """
    tracker = get_system_tracker()
    return tracker.check_system_health()


def uninstrument_system() -> None:
    """Uninstrument system component tracking."""
    global _tracker
    
    if _tracker:
        _tracker.shutdown()
        _tracker = None
        
    logger.info("AgentBay system instrumentation disabled")
