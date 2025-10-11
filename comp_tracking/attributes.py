"""
Attribute management for AgentBay system component tracking.

This module contains functions that create attributes for system resource telemetry,
providing real-time system metrics using OpenTelemetry semantic conventions.
"""

import platform
import os
import time
from typing import Any, Dict, Optional, Union, List

import psutil  # type: ignore

from .config import logger
from .resource import ResourceAttributes, SystemAttributes
from .system import get_imported_libraries


def get_system_resource_attributes() -> Dict[str, Any]:
    """
    Get real-time system resource attributes for telemetry.

    Returns:
        Dictionary containing current system information attributes
    """
    attributes: Dict[str, Any] = {
        ResourceAttributes.HOST_MACHINE: platform.machine(),
        ResourceAttributes.HOST_NAME: platform.node(),
        ResourceAttributes.HOST_NODE: platform.node(),
        ResourceAttributes.HOST_PROCESSOR: platform.processor(),
        ResourceAttributes.HOST_SYSTEM: platform.system(),
        ResourceAttributes.HOST_VERSION: platform.version(),
        ResourceAttributes.HOST_OS_RELEASE: platform.release(),
        ResourceAttributes.HOST_ARCHITECTURE: platform.architecture()[0],
    }

    # Add CPU stats
    try:
        attributes[ResourceAttributes.CPU_COUNT] = os.cpu_count() or 0
        attributes[ResourceAttributes.CPU_COUNT_LOGICAL] = psutil.cpu_count(logical=True) or 0
        attributes[ResourceAttributes.CPU_COUNT_PHYSICAL] = psutil.cpu_count(logical=False) or 0
        attributes[ResourceAttributes.CPU_PERCENT] = psutil.cpu_percent(interval=0.1)
        
        # CPU frequency if available
        try:
            freq = psutil.cpu_freq()
            if freq:
                attributes[ResourceAttributes.CPU_FREQ_CURRENT] = freq.current
                attributes[ResourceAttributes.CPU_FREQ_MIN] = freq.min
                attributes[ResourceAttributes.CPU_FREQ_MAX] = freq.max
        except (AttributeError, OSError):
            pass
            
    except Exception as e:
        logger.debug(f"Error getting CPU stats: {e}")

    # Add memory stats
    try:
        memory = psutil.virtual_memory()
        attributes[ResourceAttributes.MEMORY_TOTAL] = memory.total
        attributes[ResourceAttributes.MEMORY_AVAILABLE] = memory.available
        attributes[ResourceAttributes.MEMORY_USED] = memory.used
        attributes[ResourceAttributes.MEMORY_FREE] = memory.free
        attributes[ResourceAttributes.MEMORY_PERCENT] = memory.percent
        
        # Add buffers/cached if available (Linux)
        if hasattr(memory, 'buffers'):
            attributes[ResourceAttributes.MEMORY_BUFFERS] = memory.buffers
        if hasattr(memory, 'cached'):
            attributes[ResourceAttributes.MEMORY_CACHED] = memory.cached
            
    except Exception as e:
        logger.debug(f"Error getting memory stats: {e}")

    # Add swap stats
    try:
        swap = psutil.swap_memory()
        attributes[ResourceAttributes.SWAP_TOTAL] = swap.total
        attributes[ResourceAttributes.SWAP_USED] = swap.used
        attributes[ResourceAttributes.SWAP_FREE] = swap.free
        attributes[ResourceAttributes.SWAP_PERCENT] = swap.percent
    except Exception as e:
        logger.debug(f"Error getting swap stats: {e}")

    # Add disk stats for root partition
    try:
        disk_usage = psutil.disk_usage('/')
        attributes[ResourceAttributes.DISK_TOTAL] = disk_usage.total
        attributes[ResourceAttributes.DISK_USED] = disk_usage.used
        attributes[ResourceAttributes.DISK_FREE] = disk_usage.free
        attributes[ResourceAttributes.DISK_PERCENT] = (disk_usage.used / disk_usage.total) * 100
    except Exception as e:
        logger.debug(f"Error getting disk stats: {e}")

    # Add system uptime
    try:
        boot_time = psutil.boot_time()
        attributes[ResourceAttributes.SYSTEM_UPTIME] = time.time() - boot_time
        attributes[ResourceAttributes.SYSTEM_BOOT_TIME] = boot_time
    except Exception as e:
        logger.debug(f"Error getting system uptime: {e}")

    return attributes


def get_process_resource_attributes() -> Dict[str, Any]:
    """
    Get current process resource attributes.
    
    Returns:
        Dictionary containing current process resource usage
    """
    attributes: Dict[str, Any] = {}
    
    try:
        process = psutil.Process()
        
        attributes[ResourceAttributes.PROCESS_PID] = process.pid
        attributes[ResourceAttributes.PROCESS_NAME] = process.name()
        attributes[ResourceAttributes.PROCESS_CPU_PERCENT] = process.cpu_percent()
        attributes[ResourceAttributes.PROCESS_MEMORY_PERCENT] = process.memory_percent()
        
        # Memory info
        memory_info = process.memory_info()
        attributes[ResourceAttributes.PROCESS_MEMORY_RSS] = memory_info.rss
        attributes[ResourceAttributes.PROCESS_MEMORY_VMS] = memory_info.vms
        
        # Additional memory info if available
        if hasattr(memory_info, 'shared'):
            attributes[ResourceAttributes.PROCESS_MEMORY_SHARED] = memory_info.shared
            
        # Process times
        cpu_times = process.cpu_times()
        attributes[ResourceAttributes.PROCESS_CPU_USER] = cpu_times.user
        attributes[ResourceAttributes.PROCESS_CPU_SYSTEM] = cpu_times.system
        
        # Process status
        attributes[ResourceAttributes.PROCESS_STATUS] = process.status()
        attributes[ResourceAttributes.PROCESS_CREATE_TIME] = process.create_time()
        
        # Thread count
        attributes[ResourceAttributes.PROCESS_THREADS] = process.num_threads()
        
    except Exception as e:
        logger.debug(f"Error getting process attributes: {e}")
        
    return attributes


def get_network_resource_attributes() -> Dict[str, Any]:
    """
    Get network resource attributes.
    
    Returns:
        Dictionary containing network interface statistics
    """
    attributes: Dict[str, Any] = {}
    
    try:
        # Get network I/O counters
        net_io = psutil.net_io_counters()
        if net_io:
            attributes[ResourceAttributes.NETWORK_BYTES_SENT] = net_io.bytes_sent
            attributes[ResourceAttributes.NETWORK_BYTES_RECV] = net_io.bytes_recv
            attributes[ResourceAttributes.NETWORK_PACKETS_SENT] = net_io.packets_sent
            attributes[ResourceAttributes.NETWORK_PACKETS_RECV] = net_io.packets_recv
            
        # Get connection count
        connections = psutil.net_connections()
        attributes[ResourceAttributes.NETWORK_CONNECTIONS] = len(connections)
        
    except Exception as e:
        logger.debug(f"Error getting network attributes: {e}")
        
    return attributes


def get_global_resource_attributes(
    service_name: str,
    project_id: Optional[str] = None,
    include_system: bool = True,
    include_process: bool = True,
    include_network: bool = False,
) -> Dict[str, Any]:
    """
    Get all global resource attributes for telemetry.

    Combines service metadata, system resources, and imported libraries.

    Args:
        service_name: Name of the service
        project_id: Optional project ID
        include_system: Include system resource attributes
        include_process: Include process resource attributes
        include_network: Include network resource attributes

    Returns:
        Dictionary containing all resource attributes
    """
    # Start with service attributes
    attributes: Dict[str, Any] = {
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SDK_NAME: "agentbay",
        ResourceAttributes.SDK_VERSION: "1.0.0",  # TODO: Get from package metadata
    }

    if project_id:
        attributes[ResourceAttributes.PROJECT_ID] = project_id

    # Add imported libraries
    if imported_libraries := get_imported_libraries():
        attributes[ResourceAttributes.IMPORTED_LIBRARIES] = imported_libraries

    # Add system attributes
    if include_system:
        try:
            system_attrs = get_system_resource_attributes()
            attributes.update(system_attrs)
        except Exception as e:
            logger.debug(f"Error getting system attributes: {e}")

    # Add process attributes
    if include_process:
        try:
            process_attrs = get_process_resource_attributes()
            attributes.update(process_attrs)
        except Exception as e:
            logger.debug(f"Error getting process attributes: {e}")

    # Add network attributes
    if include_network:
        try:
            network_attrs = get_network_resource_attributes()
            attributes.update(network_attrs)
        except Exception as e:
            logger.debug(f"Error getting network attributes: {e}")

    return attributes


def get_trace_attributes(tags: Optional[Union[Dict[str, Any], List[str]]] = None) -> Dict[str, Any]:
    """
    Get attributes for trace spans.

    Args:
        tags: Optional tags to include (dict or list)

    Returns:
        Dictionary containing trace attributes
    """
    attributes: Dict[str, Any] = {}

    if tags:
        if isinstance(tags, list):
            attributes[SystemAttributes.TAGS] = tags
        elif isinstance(tags, dict):
            attributes.update(tags)  # Add dict tags directly
        else:
            logger.warning(f"Invalid tags format: {tags}. Must be list or dict.")

    return attributes


def get_span_attributes(
    operation_name: str, 
    span_kind: str, 
    version: Optional[int] = None, 
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Get attributes for operation spans.

    Args:
        operation_name: Name of the operation being traced
        span_kind: Type of operation (from SpanKind)
        version: Optional version identifier for the operation
        **kwargs: Additional attributes to include

    Returns:
        Dictionary containing span attributes
    """
    attributes: Dict[str, Any] = {
        SystemAttributes.SPAN_KIND: span_kind,
        SystemAttributes.OPERATION_NAME: operation_name,
    }

    if version is not None:
        attributes[SystemAttributes.OPERATION_VERSION] = version

    # Add any additional attributes passed as kwargs
    attributes.update(kwargs)

    return attributes


def get_system_health_attributes() -> Dict[str, Any]:
    """
    Get system health indicators.
    
    Returns:
        Dictionary containing system health metrics
    """
    attributes: Dict[str, Any] = {}
    
    try:
        # CPU load average (Unix systems)
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()
            attributes[ResourceAttributes.SYSTEM_LOAD_1M] = load_avg[0]
            attributes[ResourceAttributes.SYSTEM_LOAD_5M] = load_avg[1]
            attributes[ResourceAttributes.SYSTEM_LOAD_15M] = load_avg[2]
            
        # Memory pressure indicator
        memory = psutil.virtual_memory()
        attributes[ResourceAttributes.MEMORY_PRESSURE] = memory.percent > 80
        
        # Disk pressure indicator
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            attributes[ResourceAttributes.DISK_PRESSURE] = disk_percent > 90
        except Exception:
            pass
            
        # Temperature if available (Linux)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            attributes[f"{ResourceAttributes.TEMPERATURE_PREFIX}.{name}"] = entry.current
                            break
        except (AttributeError, OSError):
            pass
            
    except Exception as e:
        logger.debug(f"Error getting system health attributes: {e}")
        
    return attributes
