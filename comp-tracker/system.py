"""System component tracking for AgentBay SDK.

Tracks system resources including:
- CPU usage and core information
- Memory (RAM) usage statistics
- Disk usage across all partitions
- Operating system details
- Installed packages and environment info
"""

import importlib.metadata
import os
import platform
import socket
import sys
from typing import Dict, Any, List

import psutil  # type: ignore

from .config import logger


def get_imported_libraries() -> List[str]:
    """
    Get the top-level imported libraries in the current script.

    Returns:
        List of imported library names
    """
    user_libs = []

    builtin_modules = {
        "builtins", "sys", "os", "_thread", "abc", "io", "re", "types",
        "collections", "enum", "math", "datetime", "time", "warnings",
    }

    try:
        main_module = sys.modules.get("__main__")
        if main_module and hasattr(main_module, "__dict__"):
            for name, obj in main_module.__dict__.items():
                if isinstance(obj, type(sys)) and hasattr(obj, "__name__"):
                    mod_name = obj.__name__.split(".")[0]
                    if mod_name and not mod_name.startswith("_") and mod_name not in builtin_modules:
                        user_libs.append(mod_name)
    except Exception as e:
        logger.debug(f"Error getting imports: {e}")

    return user_libs


def get_agentbay_version() -> str:
    """Get AgentBay SDK version."""
    try:
        return importlib.metadata.version("agentbay")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_sdk_details() -> Dict[str, Any]:
    """Get SDK and Python environment details."""
    try:
        return {
            "AgentBay SDK Version": get_agentbay_version(),
            "Python Version": platform.python_version(),
            "System Packages": get_sys_packages(),
        }
    except Exception as e:
        logger.debug(f"Error getting SDK details: {e}")
        return {}


def get_sys_packages() -> Dict[str, str]:
    """Get system packages and their versions."""
    sys_packages = {}
    for module in sys.modules:
        try:
            version = importlib.metadata.version(module)
            sys_packages[module] = version
        except importlib.metadata.PackageNotFoundError:
            # Skip built-in modules and those without package metadata
            continue
    return sys_packages


def get_installed_packages() -> Dict[str, Any]:
    """Get all installed packages and their versions."""
    try:
        return {
            "Installed Packages": {
                dist.metadata.get("Name"): dist.metadata.get("Version") 
                for dist in importlib.metadata.distributions()
            }
        }
    except Exception as e:
        logger.debug(f"Error getting installed packages: {e}")
        return {}


def get_current_directory() -> Dict[str, str]:
    """Get current working directory."""
    try:
        return {"Project Working Directory": os.getcwd()}
    except Exception as e:
        logger.debug(f"Error getting current directory: {e}")
        return {}


def get_virtual_env() -> Dict[str, Any]:
    """Get virtual environment information."""
    try:
        return {"Virtual Environment": os.environ.get("VIRTUAL_ENV", None)}
    except Exception as e:
        logger.debug(f"Error getting virtual env: {e}")
        return {}


def get_os_details() -> Dict[str, str]:
    """Get operating system details."""
    try:
        return {
            "Hostname": socket.gethostname(),
            "OS": platform.system(),
            "OS Version": platform.version(),
            "OS Release": platform.release(),
            "Architecture": platform.architecture()[0],
            "Machine": platform.machine(),
            "Processor": platform.processor(),
        }
    except Exception as e:
        logger.debug(f"Error getting OS details: {e}")
        return {}


def get_cpu_details() -> Dict[str, Any]:
    """Get CPU information and current usage."""
    try:
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "CPU Usage": f"{psutil.cpu_percent(interval=0.1)}%",
        }
        
        # Try to get CPU frequency info
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info["Current Frequency"] = f"{freq.current:.2f}MHz"
                cpu_info["Min Frequency"] = f"{freq.min:.2f}MHz"
                cpu_info["Max Frequency"] = f"{freq.max:.2f}MHz"
        except (AttributeError, OSError):
            # CPU frequency not available on all platforms
            pass
            
        # Get per-CPU usage
        try:
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
            cpu_info["Per CPU Usage"] = [f"{usage}%" for usage in per_cpu]
        except Exception:
            pass
            
        return cpu_info
    except Exception as e:
        logger.debug(f"Error getting CPU details: {e}")
        return {}


def get_ram_details() -> Dict[str, Any]:
    """Get memory (RAM) usage details."""
    try:
        ram_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        return {
            "Total": f"{ram_info.total / (1024**3):.2f} GB",
            "Available": f"{ram_info.available / (1024**3):.2f} GB",
            "Used": f"{ram_info.used / (1024**3):.2f} GB",
            "Percentage": f"{ram_info.percent}%",
            "Free": f"{ram_info.free / (1024**3):.2f} GB",
            "Buffers": f"{getattr(ram_info, 'buffers', 0) / (1024**3):.2f} GB",
            "Cached": f"{getattr(ram_info, 'cached', 0) / (1024**3):.2f} GB",
            "Swap Total": f"{swap_info.total / (1024**3):.2f} GB",
            "Swap Used": f"{swap_info.used / (1024**3):.2f} GB",
            "Swap Free": f"{swap_info.free / (1024**3):.2f} GB",
            "Swap Percentage": f"{swap_info.percent}%",
        }
    except Exception as e:
        logger.debug(f"Error getting RAM details: {e}")
        return {}


def get_disk_details() -> Dict[str, Dict[str, Any]]:
    """Get disk usage details for all partitions."""
    partitions = psutil.disk_partitions()
    disk_info = {}
    
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_info[partition.device] = {
                "Mountpoint": partition.mountpoint,
                "File System": partition.fstype,
                "Total": f"{usage.total / (1024**3):.2f} GB",
                "Used": f"{usage.used / (1024**3):.2f} GB",
                "Free": f"{usage.free / (1024**3):.2f} GB",
                "Percentage": f"{usage.percent}%",
            }
        except (OSError, PermissionError) as e:
            # Skip inaccessible partitions
            logger.debug(f"Mountpoint {partition.mountpoint} inaccessible: {e}")
            
    return disk_info


def get_network_details() -> Dict[str, Any]:
    """Get network interface details."""
    try:
        network_info = {}
        net_io = psutil.net_io_counters(pernic=True)
        
        for interface, stats in net_io.items():
            network_info[interface] = {
                "Bytes Sent": f"{stats.bytes_sent / (1024**2):.2f} MB",
                "Bytes Received": f"{stats.bytes_recv / (1024**2):.2f} MB",
                "Packets Sent": stats.packets_sent,
                "Packets Received": stats.packets_recv,
            }
            
        return network_info
    except Exception as e:
        logger.debug(f"Error getting network details: {e}")
        return {}


def get_process_details() -> Dict[str, Any]:
    """Get current process details."""
    try:
        process = psutil.Process()
        return {
            "PID": process.pid,
            "Name": process.name(),
            "CPU Percent": f"{process.cpu_percent()}%",
            "Memory Percent": f"{process.memory_percent():.2f}%",
            "Memory Info": f"{process.memory_info().rss / (1024**2):.2f} MB",
            "Create Time": process.create_time(),
            "Status": process.status(),
        }
    except Exception as e:
        logger.debug(f"Error getting process details: {e}")
        return {}


def get_host_env(opt_out: bool = False) -> Dict[str, Any]:
    """
    Get comprehensive host environment information.
    
    Args:
        opt_out: If True, collect minimal data (privacy mode)
        
    Returns:
        Dictionary containing system information
    """
    if opt_out:
        return {
            "SDK": get_sdk_details(),
            "OS": get_os_details(),
            "Project Working Directory": get_current_directory(),
            "Virtual Environment": get_virtual_env(),
        }
    else:
        return {
            "SDK": get_sdk_details(),
            "OS": get_os_details(),
            "CPU": get_cpu_details(),
            "RAM": get_ram_details(),
            "Disk": get_disk_details(),
            "Network": get_network_details(),
            "Process": get_process_details(),
            "Installed Packages": get_installed_packages(),
            "Project Working Directory": get_current_directory(),
            "Virtual Environment": get_virtual_env(),
            "Imported Libraries": get_imported_libraries(),
        }


def get_system_snapshot() -> Dict[str, Any]:
    """Get a complete snapshot of system resources at current moment."""
    return {
        "timestamp": psutil.boot_time(),
        "uptime_seconds": psutil.boot_time(),
        "load_average": getattr(os, 'getloadavg', lambda: (0, 0, 0))(),
        "cpu": get_cpu_details(),
        "memory": get_ram_details(),
        "disk": get_disk_details(),
        "network": get_network_details(),
        "process": get_process_details(),
    }
