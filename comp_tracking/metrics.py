"""
System metrics collection and recording for AgentBay.

This module provides functions to record system resource metrics using OpenTelemetry,
including CPU usage, memory consumption, disk usage, and network statistics.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable

from opentelemetry import metrics
from opentelemetry.metrics import Meter, Counter, Gauge, Histogram

import psutil  # type: ignore

from .config import logger
from .resource import ResourceAttributes, MetricAttributes
from .attributes import (
    get_system_resource_attributes,
    get_process_resource_attributes,
    get_network_resource_attributes,
)


class SystemMetricsRecorder:
    """Records system metrics using OpenTelemetry."""
    
    def __init__(self, meter_name: str = "agentbay.system"):
        self.meter: Meter = metrics.get_meter(meter_name)
        self._setup_instruments()
        self._last_network_stats: Optional[Dict[str, Any]] = None
        self._last_network_time: Optional[float] = None
        
    def _setup_instruments(self) -> None:
        """Setup OpenTelemetry instruments for system metrics."""
        
        # CPU metrics
        self.cpu_usage_gauge = self.meter.create_gauge(
            name="system_cpu_usage_percent",
            description="CPU usage percentage",
            unit=MetricAttributes.PERCENT,
        )
        
        self.cpu_count_gauge = self.meter.create_gauge(
            name="system_cpu_count",
            description="Number of CPU cores",
            unit=MetricAttributes.COUNT,
        )
        
        # Memory metrics
        self.memory_usage_gauge = self.meter.create_gauge(
            name="system_memory_usage_bytes",
            description="Memory usage in bytes",
            unit=MetricAttributes.BYTES,
        )
        
        self.memory_usage_percent_gauge = self.meter.create_gauge(
            name="system_memory_usage_percent",
            description="Memory usage percentage",
            unit=MetricAttributes.PERCENT,
        )
        
        self.memory_available_gauge = self.meter.create_gauge(
            name="system_memory_available_bytes",
            description="Available memory in bytes",
            unit=MetricAttributes.BYTES,
        )
        
        # Disk metrics
        self.disk_usage_gauge = self.meter.create_gauge(
            name="system_disk_usage_bytes",
            description="Disk usage in bytes",
            unit=MetricAttributes.BYTES,
        )
        
        self.disk_usage_percent_gauge = self.meter.create_gauge(
            name="system_disk_usage_percent",
            description="Disk usage percentage",
            unit=MetricAttributes.PERCENT,
        )
        
        # Network metrics
        self.network_bytes_counter = self.meter.create_counter(
            name="system_network_bytes_total",
            description="Total network bytes transferred",
            unit=MetricAttributes.BYTES,
        )
        
        self.network_packets_counter = self.meter.create_counter(
            name="system_network_packets_total",
            description="Total network packets transferred",
            unit=MetricAttributes.COUNT,
        )
        
        # Process metrics
        self.process_cpu_usage_gauge = self.meter.create_gauge(
            name="process_cpu_usage_percent",
            description="Process CPU usage percentage",
            unit=MetricAttributes.PERCENT,
        )
        
        self.process_memory_gauge = self.meter.create_gauge(
            name="process_memory_usage_bytes",
            description="Process memory usage in bytes",
            unit=MetricAttributes.BYTES,
        )
        
        # System health metrics
        self.system_uptime_gauge = self.meter.create_gauge(
            name="system_uptime_seconds",
            description="System uptime in seconds",
            unit=MetricAttributes.SECONDS,
        )
        
        self.system_load_gauge = self.meter.create_gauge(
            name="system_load_average",
            description="System load average",
            unit=MetricAttributes.COUNT,
        )

    def record_cpu_metrics(self) -> None:
        """Record CPU-related metrics."""
        try:
            # CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage_gauge.set(cpu_percent)
            
            # CPU count
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count:
                self.cpu_count_gauge.set(cpu_count)
                
            # Per-CPU usage
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
            for i, usage in enumerate(per_cpu):
                self.cpu_usage_gauge.set(usage, {"cpu": str(i)})
                
        except Exception as e:
            logger.debug(f"Error recording CPU metrics: {e}")

    def record_memory_metrics(self) -> None:
        """Record memory-related metrics."""
        try:
            memory = psutil.virtual_memory()
            
            # Memory usage
            self.memory_usage_gauge.set(memory.used)
            self.memory_usage_percent_gauge.set(memory.percent)
            self.memory_available_gauge.set(memory.available)
            
            # Additional memory metrics with labels
            self.memory_usage_gauge.set(memory.total, {"type": "total"})
            self.memory_usage_gauge.set(memory.free, {"type": "free"})
            
            if hasattr(memory, 'buffers'):
                self.memory_usage_gauge.set(memory.buffers, {"type": "buffers"})
            if hasattr(memory, 'cached'):
                self.memory_usage_gauge.set(memory.cached, {"type": "cached"})
                
            # Swap memory
            swap = psutil.swap_memory()
            self.memory_usage_gauge.set(swap.total, {"type": "swap_total"})
            self.memory_usage_gauge.set(swap.used, {"type": "swap_used"})
            self.memory_usage_gauge.set(swap.free, {"type": "swap_free"})
            
        except Exception as e:
            logger.debug(f"Error recording memory metrics: {e}")

    def record_disk_metrics(self) -> None:
        """Record disk-related metrics."""
        try:
            # Get disk usage for all partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    labels = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                    }
                    
                    self.disk_usage_gauge.set(usage.total, {**labels, "type": "total"})
                    self.disk_usage_gauge.set(usage.used, {**labels, "type": "used"})
                    self.disk_usage_gauge.set(usage.free, {**labels, "type": "free"})
                    
                    usage_percent = (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    self.disk_usage_percent_gauge.set(usage_percent, labels)
                    
                except (OSError, PermissionError):
                    # Skip inaccessible partitions
                    continue
                    
        except Exception as e:
            logger.debug(f"Error recording disk metrics: {e}")

    def record_network_metrics(self) -> None:
        """Record network-related metrics."""
        try:
            current_time = time.time()
            net_io = psutil.net_io_counters()
            
            if net_io:
                # Record absolute values
                self.network_bytes_counter.add(0, {"direction": "sent", "value": str(net_io.bytes_sent)})
                self.network_bytes_counter.add(0, {"direction": "received", "value": str(net_io.bytes_recv)})
                self.network_packets_counter.add(0, {"direction": "sent", "value": str(net_io.packets_sent)})
                self.network_packets_counter.add(0, {"direction": "received", "value": str(net_io.packets_recv)})
                
                # Calculate rates if we have previous data
                if self._last_network_stats and self._last_network_time:
                    time_delta = current_time - self._last_network_time
                    if time_delta > 0:
                        bytes_sent_rate = (net_io.bytes_sent - self._last_network_stats["bytes_sent"]) / time_delta
                        bytes_recv_rate = (net_io.bytes_recv - self._last_network_stats["bytes_recv"]) / time_delta
                        
                        # Record rates as gauges
                        self.meter.create_gauge("system_network_bytes_per_second", 
                                              description="Network bytes per second").set(
                            bytes_sent_rate, {"direction": "sent"}
                        )
                        self.meter.create_gauge("system_network_bytes_per_second",
                                              description="Network bytes per second").set(
                            bytes_recv_rate, {"direction": "received"}
                        )
                
                # Store current stats for next calculation
                self._last_network_stats = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                self._last_network_time = current_time
                
        except Exception as e:
            logger.debug(f"Error recording network metrics: {e}")

    def record_process_metrics(self) -> None:
        """Record process-related metrics."""
        try:
            process = psutil.Process()
            
            # Process CPU and memory
            self.process_cpu_usage_gauge.set(process.cpu_percent())
            
            memory_info = process.memory_info()
            self.process_memory_gauge.set(memory_info.rss, {"type": "rss"})
            self.process_memory_gauge.set(memory_info.vms, {"type": "vms"})
            
            # Process metadata
            labels = {
                "pid": str(process.pid),
                "name": process.name(),
                "status": process.status(),
            }
            
            self.meter.create_gauge("process_threads_count",
                                  description="Number of process threads").set(
                process.num_threads(), labels
            )
            
        except Exception as e:
            logger.debug(f"Error recording process metrics: {e}")

    def record_system_health_metrics(self) -> None:
        """Record system health metrics."""
        try:
            # System uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            self.system_uptime_gauge.set(uptime)
            
            # Load average (Unix systems)
            try:
                import os
                if hasattr(os, 'getloadavg'):
                    load_avg = os.getloadavg()
                    self.system_load_gauge.set(load_avg[0], {"period": "1m"})
                    self.system_load_gauge.set(load_avg[1], {"period": "5m"})
                    self.system_load_gauge.set(load_avg[2], {"period": "15m"})
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Error recording system health metrics: {e}")

    def record_all_metrics(self) -> None:
        """Record all system metrics."""
        self.record_cpu_metrics()
        self.record_memory_metrics()
        self.record_disk_metrics()
        self.record_network_metrics()
        self.record_process_metrics()
        self.record_system_health_metrics()


class PeriodicMetricsCollector:
    """Collects system metrics periodically in a background thread."""
    
    def __init__(self, recorder: SystemMetricsRecorder, interval: float = 30.0):
        self.recorder = recorder
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
    def start(self) -> None:
        """Start periodic metrics collection."""
        if self._running:
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.debug(f"Started periodic metrics collection (interval: {self.interval}s)")
        
    def stop(self) -> None:
        """Stop periodic metrics collection."""
        if not self._running:
            return
            
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._running = False
        logger.debug("Stopped periodic metrics collection")
        
    def _collect_loop(self) -> None:
        """Main collection loop."""
        while not self._stop_event.wait(self.interval):
            try:
                self.recorder.record_all_metrics()
            except Exception as e:
                logger.error(f"Error during metrics collection: {e}")


# Global instances
_recorder: Optional[SystemMetricsRecorder] = None
_collector: Optional[PeriodicMetricsCollector] = None


def get_metrics_recorder() -> SystemMetricsRecorder:
    """Get the global metrics recorder instance."""
    global _recorder
    if _recorder is None:
        _recorder = SystemMetricsRecorder()
    return _recorder


def start_periodic_collection(interval: float = 30.0) -> None:
    """Start periodic system metrics collection."""
    global _collector
    if _collector is None:
        recorder = get_metrics_recorder()
        _collector = PeriodicMetricsCollector(recorder, interval)
    _collector.start()


def stop_periodic_collection() -> None:
    """Stop periodic system metrics collection."""
    global _collector
    if _collector:
        _collector.stop()


def record_system_snapshot() -> None:
    """Record a one-time snapshot of all system metrics."""
    recorder = get_metrics_recorder()
    recorder.record_all_metrics()


# Convenience functions for specific metrics
def record_cpu_usage() -> None:
    """Record CPU usage metrics."""
    get_metrics_recorder().record_cpu_metrics()


def record_memory_usage() -> None:
    """Record memory usage metrics."""
    get_metrics_recorder().record_memory_metrics()


def record_disk_usage() -> None:
    """Record disk usage metrics."""
    get_metrics_recorder().record_disk_metrics()


def record_network_stats() -> None:
    """Record network statistics."""
    get_metrics_recorder().record_network_metrics()


def record_process_stats() -> None:
    """Record process statistics."""
    get_metrics_recorder().record_process_metrics()
