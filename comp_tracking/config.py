"""Configuration and settings for AgentBay system component tracking.

This module provides configuration management for system monitoring,
including privacy settings, collection intervals, and alert thresholds.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Union, List


# Setup logger for the component tracker
logger = logging.getLogger("agentbay.comp_tracker")
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class SystemTrackingConfig:
    """Configuration for system component tracking."""
    
    # Collection settings
    enable_cpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_disk_monitoring: bool = True
    enable_network_monitoring: bool = False  # Disabled by default for privacy
    enable_process_monitoring: bool = True
    enable_system_health: bool = True
    
    # Collection intervals (in seconds)
    cpu_collection_interval: float = 30.0
    memory_collection_interval: float = 30.0
    disk_collection_interval: float = 60.0
    network_collection_interval: float = 60.0
    process_collection_interval: float = 30.0
    system_health_interval: float = 60.0
    
    # Default periodic collection interval
    default_collection_interval: float = 30.0
    
    # Alert thresholds (percentages)
    cpu_alert_threshold: float = 80.0
    memory_alert_threshold: float = 85.0
    disk_alert_threshold: float = 90.0
    
    # Privacy settings
    privacy_mode: bool = False
    capture_process_names: bool = True
    capture_network_details: bool = False
    anonymize_hostnames: bool = False
    
    # Data retention
    data_retention_days: int = 30
    
    # Export settings
    enable_metrics_export: bool = True
    enable_traces_export: bool = True
    
    # Redaction function
    redactor: Optional[Callable[[str], str]] = None


# Global configuration instance
_config = SystemTrackingConfig()


def configure(
    enable_cpu_monitoring: Optional[bool] = None,
    enable_memory_monitoring: Optional[bool] = None,
    enable_disk_monitoring: Optional[bool] = None,
    enable_network_monitoring: Optional[bool] = None,
    enable_process_monitoring: Optional[bool] = None,
    enable_system_health: Optional[bool] = None,
    cpu_collection_interval: Optional[float] = None,
    memory_collection_interval: Optional[float] = None,
    disk_collection_interval: Optional[float] = None,
    network_collection_interval: Optional[float] = None,
    process_collection_interval: Optional[float] = None,
    system_health_interval: Optional[float] = None,
    default_collection_interval: Optional[float] = None,
    cpu_alert_threshold: Optional[float] = None,
    memory_alert_threshold: Optional[float] = None,
    disk_alert_threshold: Optional[float] = None,
    privacy_mode: Optional[bool] = None,
    capture_process_names: Optional[bool] = None,
    capture_network_details: Optional[bool] = None,
    anonymize_hostnames: Optional[bool] = None,
    data_retention_days: Optional[int] = None,
    enable_metrics_export: Optional[bool] = None,
    enable_traces_export: Optional[bool] = None,
    redactor: Optional[Callable[[str], str]] = None,
) -> None:
    """
    Configure system tracking settings.
    
    Args:
        enable_cpu_monitoring: Enable CPU monitoring
        enable_memory_monitoring: Enable memory monitoring
        enable_disk_monitoring: Enable disk monitoring
        enable_network_monitoring: Enable network monitoring
        enable_process_monitoring: Enable process monitoring
        enable_system_health: Enable system health monitoring
        cpu_collection_interval: CPU collection interval in seconds
        memory_collection_interval: Memory collection interval in seconds
        disk_collection_interval: Disk collection interval in seconds
        network_collection_interval: Network collection interval in seconds
        process_collection_interval: Process collection interval in seconds
        system_health_interval: System health collection interval in seconds
        default_collection_interval: Default collection interval in seconds
        cpu_alert_threshold: CPU usage alert threshold (percentage)
        memory_alert_threshold: Memory usage alert threshold (percentage)
        disk_alert_threshold: Disk usage alert threshold (percentage)
        privacy_mode: Enable privacy mode (minimal data collection)
        capture_process_names: Capture process names
        capture_network_details: Capture detailed network information
        anonymize_hostnames: Anonymize hostname in data
        data_retention_days: Data retention period in days
        enable_metrics_export: Enable metrics export
        enable_traces_export: Enable traces export
        redactor: Function to redact sensitive data
    """
    global _config
    
    if enable_cpu_monitoring is not None:
        _config.enable_cpu_monitoring = enable_cpu_monitoring
    if enable_memory_monitoring is not None:
        _config.enable_memory_monitoring = enable_memory_monitoring
    if enable_disk_monitoring is not None:
        _config.enable_disk_monitoring = enable_disk_monitoring
    if enable_network_monitoring is not None:
        _config.enable_network_monitoring = enable_network_monitoring
    if enable_process_monitoring is not None:
        _config.enable_process_monitoring = enable_process_monitoring
    if enable_system_health is not None:
        _config.enable_system_health = enable_system_health
        
    if cpu_collection_interval is not None:
        _config.cpu_collection_interval = cpu_collection_interval
    if memory_collection_interval is not None:
        _config.memory_collection_interval = memory_collection_interval
    if disk_collection_interval is not None:
        _config.disk_collection_interval = disk_collection_interval
    if network_collection_interval is not None:
        _config.network_collection_interval = network_collection_interval
    if process_collection_interval is not None:
        _config.process_collection_interval = process_collection_interval
    if system_health_interval is not None:
        _config.system_health_interval = system_health_interval
    if default_collection_interval is not None:
        _config.default_collection_interval = default_collection_interval
        
    if cpu_alert_threshold is not None:
        _config.cpu_alert_threshold = cpu_alert_threshold
    if memory_alert_threshold is not None:
        _config.memory_alert_threshold = memory_alert_threshold
    if disk_alert_threshold is not None:
        _config.disk_alert_threshold = disk_alert_threshold
        
    if privacy_mode is not None:
        _config.privacy_mode = privacy_mode
    if capture_process_names is not None:
        _config.capture_process_names = capture_process_names
    if capture_network_details is not None:
        _config.capture_network_details = capture_network_details
    if anonymize_hostnames is not None:
        _config.anonymize_hostnames = anonymize_hostnames
    if data_retention_days is not None:
        _config.data_retention_days = data_retention_days
        
    if enable_metrics_export is not None:
        _config.enable_metrics_export = enable_metrics_export
    if enable_traces_export is not None:
        _config.enable_traces_export = enable_traces_export
        
    if redactor is not None:
        _config.redactor = redactor


def get_config() -> SystemTrackingConfig:
    """Get the current system tracking configuration."""
    return _config


def configure_from_env() -> None:
    """Configure system tracking from environment variables."""
    
    # Monitoring enables
    if os.getenv("AGENTBAY_ENABLE_CPU_MONITORING"):
        _config.enable_cpu_monitoring = os.getenv("AGENTBAY_ENABLE_CPU_MONITORING", "true").lower() == "true"
    if os.getenv("AGENTBAY_ENABLE_MEMORY_MONITORING"):
        _config.enable_memory_monitoring = os.getenv("AGENTBAY_ENABLE_MEMORY_MONITORING", "true").lower() == "true"
    if os.getenv("AGENTBAY_ENABLE_DISK_MONITORING"):
        _config.enable_disk_monitoring = os.getenv("AGENTBAY_ENABLE_DISK_MONITORING", "true").lower() == "true"
    if os.getenv("AGENTBAY_ENABLE_NETWORK_MONITORING"):
        _config.enable_network_monitoring = os.getenv("AGENTBAY_ENABLE_NETWORK_MONITORING", "false").lower() == "true"
    if os.getenv("AGENTBAY_ENABLE_PROCESS_MONITORING"):
        _config.enable_process_monitoring = os.getenv("AGENTBAY_ENABLE_PROCESS_MONITORING", "true").lower() == "true"
    
    # Collection intervals
    if os.getenv("AGENTBAY_COLLECTION_INTERVAL"):
        try:
            _config.default_collection_interval = float(os.getenv("AGENTBAY_COLLECTION_INTERVAL", "30"))
        except ValueError:
            logger.warning("Invalid AGENTBAY_COLLECTION_INTERVAL value, using default")
    
    # Alert thresholds
    if os.getenv("AGENTBAY_CPU_THRESHOLD"):
        try:
            _config.cpu_alert_threshold = float(os.getenv("AGENTBAY_CPU_THRESHOLD", "80"))
        except ValueError:
            logger.warning("Invalid AGENTBAY_CPU_THRESHOLD value, using default")
    
    if os.getenv("AGENTBAY_MEMORY_THRESHOLD"):
        try:
            _config.memory_alert_threshold = float(os.getenv("AGENTBAY_MEMORY_THRESHOLD", "85"))
        except ValueError:
            logger.warning("Invalid AGENTBAY_MEMORY_THRESHOLD value, using default")
    
    if os.getenv("AGENTBAY_DISK_THRESHOLD"):
        try:
            _config.disk_alert_threshold = float(os.getenv("AGENTBAY_DISK_THRESHOLD", "90"))
        except ValueError:
            logger.warning("Invalid AGENTBAY_DISK_THRESHOLD value, using default")
    
    # Privacy settings
    if os.getenv("AGENTBAY_PRIVACY_MODE"):
        _config.privacy_mode = os.getenv("AGENTBAY_PRIVACY_MODE", "false").lower() == "true"
    
    if os.getenv("AGENTBAY_ANONYMIZE_HOSTNAMES"):
        _config.anonymize_hostnames = os.getenv("AGENTBAY_ANONYMIZE_HOSTNAMES", "false").lower() == "true"


def maybe_redact(text: Optional[str]) -> Optional[str]:
    """
    Apply redaction to text if configured.
    
    Args:
        text: Text to potentially redact
        
    Returns:
        Redacted text or None if privacy mode is enabled
    """
    if text is None:
        return None
        
    if _config.privacy_mode:
        return None
        
    if _config.redactor is not None:
        try:
            return _config.redactor(text)
        except Exception as e:
            logger.debug(f"Error in redactor function: {e}")
            return text
            
    return text


def should_collect_metric(metric_type: str) -> bool:
    """
    Check if a specific metric type should be collected.
    
    Args:
        metric_type: Type of metric ('cpu', 'memory', 'disk', 'network', 'process', 'health')
        
    Returns:
        True if the metric should be collected
    """
    metric_map = {
        'cpu': _config.enable_cpu_monitoring,
        'memory': _config.enable_memory_monitoring,
        'disk': _config.enable_disk_monitoring,
        'network': _config.enable_network_monitoring,
        'process': _config.enable_process_monitoring,
        'health': _config.enable_system_health,
        'system_health': _config.enable_system_health,
    }
    
    return metric_map.get(metric_type, False)


def get_collection_interval(metric_type: str) -> float:
    """
    Get the collection interval for a specific metric type.
    
    Args:
        metric_type: Type of metric
        
    Returns:
        Collection interval in seconds
    """
    interval_map = {
        'cpu': _config.cpu_collection_interval,
        'memory': _config.memory_collection_interval,
        'disk': _config.disk_collection_interval,
        'network': _config.network_collection_interval,
        'process': _config.process_collection_interval,
        'health': _config.system_health_interval,
        'system_health': _config.system_health_interval,
    }
    
    return interval_map.get(metric_type, _config.default_collection_interval)


def get_alert_threshold(metric_type: str) -> float:
    """
    Get the alert threshold for a specific metric type.
    
    Args:
        metric_type: Type of metric
        
    Returns:
        Alert threshold percentage
    """
    threshold_map = {
        'cpu': _config.cpu_alert_threshold,
        'memory': _config.memory_alert_threshold,
        'disk': _config.disk_alert_threshold,
    }
    
    return threshold_map.get(metric_type, 100.0)


def is_privacy_mode() -> bool:
    """Check if privacy mode is enabled."""
    return _config.privacy_mode


def should_capture_process_names() -> bool:
    """Check if process names should be captured."""
    return _config.capture_process_names and not _config.privacy_mode


def should_capture_network_details() -> bool:
    """Check if detailed network information should be captured."""
    return _config.capture_network_details and not _config.privacy_mode


def should_anonymize_hostnames() -> bool:
    """Check if hostnames should be anonymized."""
    return _config.anonymize_hostnames or _config.privacy_mode


# Initialize from environment on import
configure_from_env()
