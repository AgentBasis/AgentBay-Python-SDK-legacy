# AgentBay System Component Tracker - Examples

This document provides examples of how to use the AgentBay System Component Tracker to monitor CPU, memory, disk, network, and process metrics.

## Quick Start

```python
from agentbay.comp_tracking import quick_start, get_system_summary

# Start monitoring with default settings (30-second intervals)
quick_start()

# Get a quick system summary
summary = get_system_summary()
print(f"CPU: {summary['cpu_usage_percent']}%, Memory: {summary['memory_usage_percent']}%")
```

## Basic System Monitoring

```python
from agentbay.comp_tracking import instrument_system, start_system_monitoring

# Initialize system tracking
instrument_system()

# Start monitoring every 60 seconds
start_system_monitoring(interval=60.0)

# Your application code here...
```

## Configuration Options

```python
from agentbay.comp_tracking import configure, instrument_system, start_system_monitoring

# Configure monitoring settings
configure(
    enable_cpu_monitoring=True,
    enable_memory_monitoring=True,
    enable_disk_monitoring=True,
    enable_network_monitoring=True,  # Disabled by default for privacy
    enable_process_monitoring=True,
    cpu_alert_threshold=75.0,        # Alert when CPU > 75%
    memory_alert_threshold=80.0,     # Alert when memory > 80%
    disk_alert_threshold=90.0,       # Alert when disk > 90%
    privacy_mode=False,              # Set to True for minimal data collection
)

instrument_system()
start_system_monitoring()
```

## Privacy Mode

```python
from agentbay.comp_tracking import configure, quick_start

# Enable privacy mode for minimal data collection
configure(
    privacy_mode=True,
    capture_process_names=False,
    capture_network_details=False,
    anonymize_hostnames=True
)

quick_start()
```

## Manual System Information Collection

```python
from agentbay.comp_tracking import (
    get_system_info, 
    get_cpu_details, 
    get_ram_details,
    get_disk_details,
    check_system_health
)

# Get detailed system information
system_info = get_system_info(detailed=True)
print("System Info:", system_info)

# Get specific component details
cpu_info = get_cpu_details()
print("CPU Info:", cpu_info)

ram_info = get_ram_details()
print("RAM Info:", ram_info)

disk_info = get_disk_details()
print("Disk Info:", disk_info)

# Check system health
health = check_system_health()
print("System Health:", health)
```

## One-time Metrics Recording

```python
from agentbay.comp_tracking import (
    record_system_snapshot,
    record_cpu_usage,
    record_memory_usage,
    record_disk_usage
)

# Record all metrics once
record_system_snapshot()

# Record specific metrics
record_cpu_usage()
record_memory_usage()
record_disk_usage()
```

## Custom System Operation Tracing

```python
from agentbay.comp_tracking import trace_system_operation, record_system_event

# Trace a system operation
with trace_system_operation("backup_operation", backup_type="full") as span:
    # Your backup logic here
    span.set_attribute("files_backed_up", 1500)
    span.set_attribute("backup_size_gb", 25.6)

# Record a system event
record_system_event("high_memory_usage", 
                   memory_percent=87.5, 
                   threshold=85.0,
                   severity="warning")
```

## Integration with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentbay.comp_tracking import instrument_system, start_system_monitoring

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Start system monitoring
instrument_system()
start_system_monitoring()
```

## Environment Variable Configuration

```bash
# Set environment variables
export AGENTBAY_ENABLE_CPU_MONITORING=true
export AGENTBAY_ENABLE_MEMORY_MONITORING=true
export AGENTBAY_ENABLE_DISK_MONITORING=true
export AGENTBAY_ENABLE_NETWORK_MONITORING=false
export AGENTBAY_COLLECTION_INTERVAL=30
export AGENTBAY_CPU_THRESHOLD=80
export AGENTBAY_MEMORY_THRESHOLD=85
export AGENTBAY_DISK_THRESHOLD=90
export AGENTBAY_PRIVACY_MODE=false
```

```python
from agentbay.comp_tracking import configure_from_env, instrument_system, start_system_monitoring

# Configuration will be loaded from environment variables
configure_from_env()
instrument_system()
start_system_monitoring()
```

## Advanced Usage - Custom Metrics Recorder

```python
from agentbay.comp_tracking import SystemMetricsRecorder, PeriodicMetricsCollector

# Create custom metrics recorder
recorder = SystemMetricsRecorder(meter_name="my_app.system")

# Start periodic collection with custom interval
collector = PeriodicMetricsCollector(recorder, interval=15.0)
collector.start()

# Record specific metrics
recorder.record_cpu_metrics()
recorder.record_memory_metrics()

# Stop collection
collector.stop()
```

## System Health Monitoring

```python
import time
from agentbay.comp_tracking import check_system_health, record_system_event

def monitor_system_health():
    while True:
        health = check_system_health()
        
        if health["status"] == "warning":
            for alert in health["alerts"]:
                print(f"ALERT: {alert['message']}")
                record_system_event(
                    f"system_alert_{alert['type']}", 
                    **alert
                )
        
        time.sleep(60)  # Check every minute

# Run health monitoring
monitor_system_health()
```

## Integration with Existing LLM Tracking

```python
from agentbay import instrument_openai
from agentbay.comp_tracking import instrument_system, start_system_monitoring

# Initialize both LLM and system tracking
instrument_openai()
instrument_system()

# Start system monitoring
start_system_monitoring(interval=30)

# Use OpenAI as normal - both LLM and system metrics will be tracked
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Cleanup and Shutdown

```python
from agentbay.comp_tracking import stop_system_monitoring, uninstrument_system

# Stop monitoring
stop_system_monitoring()

# Uninstrument system tracking
uninstrument_system()
```

## Real-world Example: Web Application Monitoring

```python
from flask import Flask
from agentbay.comp_tracking import (
    configure, 
    instrument_system, 
    start_system_monitoring,
    trace_system_operation,
    get_system_summary
)

app = Flask(__name__)

# Configure system monitoring for web app
configure(
    enable_network_monitoring=True,
    cpu_alert_threshold=70.0,
    memory_alert_threshold=80.0,
    default_collection_interval=30.0
)

# Initialize monitoring
instrument_system(service_name="my-web-app")
start_system_monitoring()

@app.route('/health')
def health_check():
    return get_system_summary()

@app.route('/process-data')
def process_data():
    with trace_system_operation("data_processing", 
                               endpoint="/process-data") as span:
        # Your data processing logic
        span.set_attribute("records_processed", 1000)
        return {"status": "completed"}

if __name__ == '__main__':
    app.run()
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Some system metrics may require elevated permissions on certain platforms.

2. **Missing Dependencies**: Ensure `psutil` is installed: `pip install psutil`

3. **High Resource Usage**: Adjust collection intervals if monitoring impacts performance:
   ```python
   configure(default_collection_interval=120.0)  # Collect every 2 minutes
   ```

4. **Privacy Concerns**: Enable privacy mode to minimize data collection:
   ```python
   configure(privacy_mode=True)
   ```

### Debug Mode

```python
import logging
from agentbay.comp_tracking.config import logger

# Enable debug logging
logger.setLevel(logging.DEBUG)

# Now run your monitoring code
```

This covers the main usage patterns for the AgentBay System Component Tracker. The system is designed to be lightweight, configurable, and compatible with your existing LLM tracking infrastructure.
