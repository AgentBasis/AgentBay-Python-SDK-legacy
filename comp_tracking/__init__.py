"""
AgentBay System Component Tracker

A comprehensive system monitoring solution that tracks:
- CPU usage and performance metrics
- Memory (RAM) consumption and availability
- Disk usage across all partitions
- Network I/O statistics
- Process resource consumption
- System health indicators

Features:
- OpenTelemetry integration for metrics and traces
- Configurable collection intervals and privacy settings
- Real-time monitoring with background collection
- Alert thresholds for system resource usage
- Compatible with existing LLM tracking

Usage:
    # Basic system monitoring
    from agentbay.comp_tracking import instrument_system, start_system_monitoring
    
    instrument_system()
    start_system_monitoring(interval=30)  # Collect every 30 seconds
    
    # Configure monitoring options
    from agentbay.comp_tracking import configure
    
    configure(
        enable_network_monitoring=True,
        cpu_alert_threshold=75.0,
        privacy_mode=False
    )
    
    # Get system information
    from agentbay.comp_tracking import get_system_info, check_system_health
    
    system_info = get_system_info(detailed=True)
    health_status = check_system_health()
    
    # Manual metrics recording
    from agentbay.comp_tracking import record_system_snapshot
    
    record_system_snapshot()  # One-time collection
"""

# Core tracking functionality
from .tracker import (
    instrument_system,
    uninstrument_system,
    start_system_monitoring,
    stop_system_monitoring,
    trace_system_operation,
    record_system_event,
    get_system_info,
    check_system_health,
    get_system_tracker,
)

# Configuration
from .config import (
    configure,
    get_config,
    configure_from_env,
    should_collect_metric,
    get_collection_interval,
    get_alert_threshold,
    is_privacy_mode,
    should_capture_process_names,
    should_capture_network_details,
    should_anonymize_hostnames,
    SystemTrackingConfig,
)

# System information functions
from .system import (
    get_host_env,
    get_system_snapshot,
    get_cpu_details,
    get_ram_details,
    get_disk_details,
    get_network_details,
    get_process_details,
    get_os_details,
    get_sdk_details,
    get_installed_packages,
    get_current_directory,
    get_virtual_env,
    get_imported_libraries,
)

# Metrics recording
from .metrics import (
    record_system_snapshot,
    record_cpu_usage,
    record_memory_usage,
    record_disk_usage,
    record_network_stats,
    record_process_stats,
    start_periodic_collection,
    stop_periodic_collection,
    get_metrics_recorder,
    SystemMetricsRecorder,
    PeriodicMetricsCollector,
)

# Attributes and resource definitions
from .attributes import (
    get_system_resource_attributes,
    get_process_resource_attributes,
    get_network_resource_attributes,
    get_global_resource_attributes,
    get_trace_attributes,
    get_span_attributes,
    get_system_health_attributes,
)

from .resource import (
    ResourceAttributes,
    SystemAttributes,
    MetricAttributes,
    EventAttributes,
    ConfigAttributes,
    CORE_SYSTEM_ATTRIBUTES,
    PERFORMANCE_ATTRIBUTES,
    HEALTH_ATTRIBUTES,
)

# Version info
__version__ = "1.0.0"
__author__ = "AgentBay Team"
__description__ = "System component tracking for AgentBay SDK"

# Export all public functions and classes
__all__ = [
    # Core functionality
    "instrument_system",
    "uninstrument_system",
    "start_system_monitoring",
    "stop_system_monitoring",
    "trace_system_operation",
    "record_system_event",
    "get_system_info",
    "check_system_health",
    "get_system_tracker",
    
    # Configuration
    "configure",
    "get_config",
    "configure_from_env",
    "should_collect_metric",
    "get_collection_interval",
    "get_alert_threshold",
    "is_privacy_mode",
    "should_capture_process_names",
    "should_capture_network_details",
    "should_anonymize_hostnames",
    "SystemTrackingConfig",
    
    # System information
    "get_host_env",
    "get_system_snapshot",
    "get_cpu_details",
    "get_ram_details",
    "get_disk_details",
    "get_network_details",
    "get_process_details",
    "get_os_details",
    "get_sdk_details",
    "get_installed_packages",
    "get_current_directory",
    "get_virtual_env",
    "get_imported_libraries",
    
    # Metrics
    "record_system_snapshot",
    "record_cpu_usage",
    "record_memory_usage",
    "record_disk_usage",
    "record_network_stats",
    "record_process_stats",
    "start_periodic_collection",
    "stop_periodic_collection",
    "get_metrics_recorder",
    "SystemMetricsRecorder",
    "PeriodicMetricsCollector",
    
    # Attributes
    "get_system_resource_attributes",
    "get_process_resource_attributes",
    "get_network_resource_attributes",
    "get_global_resource_attributes",
    "get_trace_attributes",
    "get_span_attributes",
    "get_system_health_attributes",
    
    # Resource definitions
    "ResourceAttributes",
    "SystemAttributes",
    "MetricAttributes",
    "EventAttributes",
    "ConfigAttributes",
    "CORE_SYSTEM_ATTRIBUTES",
    "PERFORMANCE_ATTRIBUTES",
    "HEALTH_ATTRIBUTES",
]


# Convenience functions for quick setup
def quick_start(
    interval: float = 30.0,
    enable_network: bool = False,
    privacy_mode: bool = False,
    **config_kwargs
) -> None:
    """
    Quick start system monitoring with common settings.
    
    Args:
        interval: Collection interval in seconds
        enable_network: Enable network monitoring
        privacy_mode: Enable privacy mode (minimal data collection)
        **config_kwargs: Additional configuration options
    """
    # Configure settings
    configure(
        enable_network_monitoring=enable_network,
        privacy_mode=privacy_mode,
        **config_kwargs
    )
    
    # Initialize and start monitoring
    instrument_system()
    start_system_monitoring(interval)


def get_system_summary() -> dict:
    """
    Get a summary of current system status.
    
    Returns:
        Dictionary with system summary information
    """
    try:
        import psutil
        
        # Get basic system info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        summary = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "status": "healthy"
        }
        
        # Add disk info for root partition
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            summary["disk_usage_percent"] = disk_percent
        except Exception:
            pass
            
        # Determine overall status
        config = get_config()
        if (cpu_percent > config.cpu_alert_threshold or 
            memory.percent > config.memory_alert_threshold):
            summary["status"] = "warning"
            
        return summary
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Add convenience aliases
monitor_system = start_system_monitoring
stop_monitoring = stop_system_monitoring
system_info = get_system_info
system_health = check_system_health
