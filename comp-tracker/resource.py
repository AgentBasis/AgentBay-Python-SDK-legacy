"""
Resource attribute semantic conventions for AgentBay system tracking.

This module defines standard resource attributes used to identify system resources
in AgentBay telemetry data, following OpenTelemetry semantic conventions where applicable.
"""


class ResourceAttributes:
    """
    Resource attributes for AgentBay system component tracking.

    These attributes provide standard identifiers for system resources being monitored.
    """

    # Project and service identifiers
    PROJECT_ID = "agentbay.project.id"
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"

    # SDK attributes
    SDK_NAME = "agentbay.sdk.name"
    SDK_VERSION = "agentbay.sdk.version"

    # Environment attributes
    ENVIRONMENT = "agentbay.environment"
    DEPLOYMENT_ENVIRONMENT = "deployment.environment"

    # Host machine attributes
    HOST_MACHINE = "host.machine"
    HOST_NAME = "host.name"
    HOST_NODE = "host.node"
    HOST_OS_RELEASE = "host.os_release"
    HOST_PROCESSOR = "host.processor"
    HOST_SYSTEM = "host.system"
    HOST_VERSION = "host.version"
    HOST_ARCHITECTURE = "host.architecture"

    # CPU attributes
    CPU_COUNT = "cpu.count"
    CPU_COUNT_LOGICAL = "cpu.count.logical"
    CPU_COUNT_PHYSICAL = "cpu.count.physical"
    CPU_PERCENT = "cpu.percent"
    CPU_FREQ_CURRENT = "cpu.frequency.current"
    CPU_FREQ_MIN = "cpu.frequency.min"
    CPU_FREQ_MAX = "cpu.frequency.max"

    # Memory attributes
    MEMORY_TOTAL = "memory.total"
    MEMORY_AVAILABLE = "memory.available"
    MEMORY_USED = "memory.used"
    MEMORY_FREE = "memory.free"
    MEMORY_PERCENT = "memory.percent"
    MEMORY_BUFFERS = "memory.buffers"
    MEMORY_CACHED = "memory.cached"
    MEMORY_PRESSURE = "memory.pressure"

    # Swap attributes
    SWAP_TOTAL = "swap.total"
    SWAP_USED = "swap.used"
    SWAP_FREE = "swap.free"
    SWAP_PERCENT = "swap.percent"

    # Disk attributes
    DISK_TOTAL = "disk.total"
    DISK_USED = "disk.used"
    DISK_FREE = "disk.free"
    DISK_PERCENT = "disk.percent"
    DISK_PRESSURE = "disk.pressure"

    # Network attributes
    NETWORK_BYTES_SENT = "network.bytes.sent"
    NETWORK_BYTES_RECV = "network.bytes.received"
    NETWORK_PACKETS_SENT = "network.packets.sent"
    NETWORK_PACKETS_RECV = "network.packets.received"
    NETWORK_CONNECTIONS = "network.connections.count"

    # Process attributes
    PROCESS_PID = "process.pid"
    PROCESS_NAME = "process.name"
    PROCESS_CPU_PERCENT = "process.cpu.percent"
    PROCESS_CPU_USER = "process.cpu.user"
    PROCESS_CPU_SYSTEM = "process.cpu.system"
    PROCESS_MEMORY_PERCENT = "process.memory.percent"
    PROCESS_MEMORY_RSS = "process.memory.rss"
    PROCESS_MEMORY_VMS = "process.memory.vms"
    PROCESS_MEMORY_SHARED = "process.memory.shared"
    PROCESS_STATUS = "process.status"
    PROCESS_CREATE_TIME = "process.create_time"
    PROCESS_THREADS = "process.threads.count"

    # System health attributes
    SYSTEM_UPTIME = "system.uptime"
    SYSTEM_BOOT_TIME = "system.boot_time"
    SYSTEM_LOAD_1M = "system.load.1m"
    SYSTEM_LOAD_5M = "system.load.5m"
    SYSTEM_LOAD_15M = "system.load.15m"

    # Temperature attributes
    TEMPERATURE_PREFIX = "temperature"

    # Libraries and packages
    IMPORTED_LIBRARIES = "imported_libraries"
    INSTALLED_PACKAGES = "installed_packages"


class SystemAttributes:
    """
    System-specific span attributes for AgentBay.
    
    These attributes are used for spans related to system operations and monitoring.
    """

    # Span identification
    SPAN_KIND = "agentbay.span.kind"
    OPERATION_NAME = "operation.name"
    OPERATION_VERSION = "operation.version"

    # System operation types
    SYSTEM_OPERATION = "system.operation"
    MONITORING_OPERATION = "monitoring.operation"
    RESOURCE_OPERATION = "resource.operation"

    # Tags and metadata
    TAGS = "agentbay.tags"
    METADATA = "agentbay.metadata"

    # Monitoring intervals
    MONITORING_INTERVAL = "monitoring.interval"
    SAMPLING_RATE = "monitoring.sampling_rate"

    # Alert thresholds
    CPU_THRESHOLD = "threshold.cpu"
    MEMORY_THRESHOLD = "threshold.memory"
    DISK_THRESHOLD = "threshold.disk"
    NETWORK_THRESHOLD = "threshold.network"


class MetricAttributes:
    """
    Metric-specific attributes for AgentBay system metrics.
    
    These attributes are used for OpenTelemetry metrics related to system monitoring.
    """

    # Metric types
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"

    # Metric categories
    SYSTEM_METRIC = "system"
    PROCESS_METRIC = "process"
    NETWORK_METRIC = "network"
    APPLICATION_METRIC = "application"

    # Metric units
    BYTES = "bytes"
    PERCENT = "percent"
    COUNT = "count"
    SECONDS = "seconds"
    HERTZ = "hertz"
    CELSIUS = "celsius"

    # Metric names
    CPU_USAGE = "system.cpu.usage"
    MEMORY_USAGE = "system.memory.usage"
    DISK_USAGE = "system.disk.usage"
    NETWORK_IO = "system.network.io"
    PROCESS_COUNT = "system.process.count"


class EventAttributes:
    """
    Event-specific attributes for AgentBay system events.
    
    These attributes are used for system events and alerts.
    """

    # Event types
    ALERT = "alert"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

    # Event categories
    SYSTEM_EVENT = "system"
    PERFORMANCE_EVENT = "performance"
    SECURITY_EVENT = "security"
    APPLICATION_EVENT = "application"

    # Event severity
    SEVERITY_LOW = "low"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_HIGH = "high"
    SEVERITY_CRITICAL = "critical"

    # Event sources
    CPU_SOURCE = "cpu"
    MEMORY_SOURCE = "memory"
    DISK_SOURCE = "disk"
    NETWORK_SOURCE = "network"
    PROCESS_SOURCE = "process"


class ConfigAttributes:
    """
    Configuration attributes for AgentBay system tracking.
    
    These attributes define configuration options for system monitoring.
    """

    # Monitoring configuration
    ENABLE_CPU_MONITORING = "monitor.cpu.enabled"
    ENABLE_MEMORY_MONITORING = "monitor.memory.enabled"
    ENABLE_DISK_MONITORING = "monitor.disk.enabled"
    ENABLE_NETWORK_MONITORING = "monitor.network.enabled"
    ENABLE_PROCESS_MONITORING = "monitor.process.enabled"

    # Collection intervals (in seconds)
    CPU_COLLECTION_INTERVAL = "monitor.cpu.interval"
    MEMORY_COLLECTION_INTERVAL = "monitor.memory.interval"
    DISK_COLLECTION_INTERVAL = "monitor.disk.interval"
    NETWORK_COLLECTION_INTERVAL = "monitor.network.interval"
    PROCESS_COLLECTION_INTERVAL = "monitor.process.interval"

    # Alert thresholds
    CPU_ALERT_THRESHOLD = "alert.cpu.threshold"
    MEMORY_ALERT_THRESHOLD = "alert.memory.threshold"
    DISK_ALERT_THRESHOLD = "alert.disk.threshold"

    # Privacy settings
    PRIVACY_MODE = "privacy.mode"
    DATA_RETENTION_DAYS = "privacy.retention.days"
    ANONYMIZE_DATA = "privacy.anonymize"


# Convenience mappings for common attribute groups
CORE_SYSTEM_ATTRIBUTES = [
    ResourceAttributes.HOST_SYSTEM,
    ResourceAttributes.HOST_NAME,
    ResourceAttributes.CPU_COUNT,
    ResourceAttributes.MEMORY_TOTAL,
]

PERFORMANCE_ATTRIBUTES = [
    ResourceAttributes.CPU_PERCENT,
    ResourceAttributes.MEMORY_PERCENT,
    ResourceAttributes.DISK_PERCENT,
    ResourceAttributes.PROCESS_CPU_PERCENT,
    ResourceAttributes.PROCESS_MEMORY_PERCENT,
]

HEALTH_ATTRIBUTES = [
    ResourceAttributes.SYSTEM_UPTIME,
    ResourceAttributes.MEMORY_PRESSURE,
    ResourceAttributes.DISK_PRESSURE,
    ResourceAttributes.SYSTEM_LOAD_1M,
    ResourceAttributes.SYSTEM_LOAD_5M,
    ResourceAttributes.SYSTEM_LOAD_15M,
]
