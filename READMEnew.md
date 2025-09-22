# AgentBay SDK - Complete AI Agent Monitoring

A comprehensive SDK for monitoring AI agents and system components, providing complete observability for your AI applications.

## Features

### 🤖 LLM Tracking (`Tracker-llm/`)
- **Multi-Provider Support**: OpenAI, Anthropic, Gemini, Grok, Watson X
- **Conversation Tracking**: Full request/response logging with privacy controls
- **Token Usage**: Input, output, and total token consumption
- **Streaming Support**: Time-to-first-token, generation time, chunk counting
- **Tool Call Monitoring**: Function calls and responses
- **Privacy First**: Configurable content redaction and opt-out options

### ⚙️ System Component Tracking (`comp-tracker/`)
- **CPU Monitoring**: Usage, core count, frequency tracking
- **Memory Tracking**: RAM usage, swap, buffers, cache
- **Disk Monitoring**: Usage across all partitions
- **Network Stats**: I/O counters, connection tracking (optional)
- **Process Monitoring**: Resource usage, thread count, status
- **System Health**: Load averages, uptime, temperature (where available)

### 🔧 Core Capabilities
- **OpenTelemetry Integration**: Industry-standard telemetry
- **Real-time Monitoring**: Background collection with configurable intervals
- **Privacy Controls**: Multiple privacy modes and data anonymization
- **Alert Thresholds**: Configurable alerts for resource usage
- **Seamless Integration**: LLM and system tracking work together
- **Easy Setup**: One-line initialization for common use cases

## Quick Start

### Simple Setup
```python
from MYSDK import quick_setup

# Initialize both LLM and system tracking
quick_setup(
    llm_providers=["openai", "anthropic"],
    system_monitoring=True,
    collection_interval=30
)
```

### Using Individual Components

#### LLM Tracking Only
```python
from MYSDK.Tracker_llm import instrument_openai, instrument_anthropic

instrument_openai()
instrument_anthropic()

# Use your LLM libraries normally - they're now tracked
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### System Monitoring Only
```python
from MYSDK.comp_tracker import instrument_system, start_system_monitoring

instrument_system()
start_system_monitoring(interval=30)  # Collect every 30 seconds
```

#### Both Together
```python
from MYSDK import (
    instrument_openai, 
    instrument_system, 
    start_system_monitoring
)

# Setup LLM tracking
instrument_openai()

# Setup system monitoring
instrument_system()
start_system_monitoring()

# Your application code here...
```

## Configuration

### Privacy Settings
```python
from MYSDK import configure_llm_privacy, configure_system

# LLM privacy
configure_llm_privacy(
    capture_content=False,  # Don't capture message content
    redactor=lambda s: s.replace("secret", "[REDACTED]")
)

# System privacy
configure_system(
    privacy_mode=True,
    capture_process_names=False,
    anonymize_hostnames=True
)
```

### Monitoring Configuration
```python
from MYSDK import configure_system

configure_system(
    enable_cpu_monitoring=True,
    enable_memory_monitoring=True,
    enable_disk_monitoring=True,
    enable_network_monitoring=False,  # Disabled by default
    cpu_alert_threshold=80.0,
    memory_alert_threshold=85.0,
    default_collection_interval=30.0
)
```

### Environment Variables
```bash
# LLM tracking
export MYSDK_LLM_CAPTURE_CONTENT=true

# System tracking
export AGENTBAY_ENABLE_CPU_MONITORING=true
export AGENTBAY_ENABLE_MEMORY_MONITORING=true
export AGENTBAY_COLLECTION_INTERVAL=30
export AGENTBAY_CPU_THRESHOLD=80
export AGENTBAY_PRIVACY_MODE=false
```

## Advanced Usage

### Custom Tracing
```python
from MYSDK import trace_system_operation

with trace_system_operation("data_processing", batch_size=1000) as span:
    # Your processing logic
    result = process_data()
    span.set_attribute("records_processed", len(result))
```

### System Health Monitoring
```python
from MYSDK import check_system_health, get_system_summary

# Quick system summary
summary = get_system_summary()
print(f"CPU: {summary['cpu_usage_percent']}%, Memory: {summary['memory_usage_percent']}%")

# Detailed health check
health = check_system_health()
if health['status'] == 'warning':
    for alert in health['alerts']:
        print(f"ALERT: {alert['message']}")
```

### OpenTelemetry Integration
```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from MYSDK import init_tracing, quick_setup

# Setup custom exporter
init_tracing(exporter="otlp", endpoint="http://localhost:4318/v1/traces")

# Initialize tracking
quick_setup(llm_providers=["openai"], system_monitoring=True)
```

## Directory Structure

```
MYSDK/
├── __init__.py                 # Main SDK exports
├── README.md                   # This file
├── integration_example.py      # Complete integration example
├── Tracker-llm/               # LLM tracking components
│   ├── __init__.py
│   ├── openai.py              # OpenAI instrumentation
│   ├── anthropic.py           # Anthropic instrumentation
│   ├── gemini.py              # Gemini instrumentation
│   ├── watsonx.py             # Watson X instrumentation
│   ├── grok.py                # Grok instrumentation
│   ├── config.py              # LLM privacy configuration
│   ├── metrics.py             # LLM metrics recording
│   ├── otel.py                # OpenTelemetry setup
│   ├── manual.py              # Manual span creation
│   └── EXAMPLES.md            # LLM tracking examples
└── comp-tracker/              # System component tracking
    ├── __init__.py
    ├── tracker.py             # Main system tracker
    ├── system.py              # System information collection
    ├── attributes.py          # Real-time system attributes
    ├── metrics.py             # System metrics recording
    ├── config.py              # System tracking configuration
    ├── resource.py            # Semantic conventions
    └── EXAMPLES.md            # System tracking examples
```

## Examples

See the following files for detailed examples:
- [`integration_example.py`](./integration_example.py) - Complete integration demo
- [`Tracker-llm/EXAMPLES.md`](./Tracker-llm/EXAMPLES.md) - LLM tracking examples
- [`comp-tracker/EXAMPLES.md`](./comp-tracker/EXAMPLES.md) - System tracking examples

## Dependencies

### Required
- `opentelemetry-sdk` - Core telemetry functionality
- `opentelemetry-api` - OpenTelemetry API
- `psutil` - System information collection
- `wrapt` - Function wrapping for instrumentation

### Optional (for specific LLM providers)
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `google-generativeai` - Google Gemini
- `ibm-watsonx-ai` - IBM Watson X

## Privacy and Security

AgentBay SDK is designed with privacy in mind:

- **Opt-in by Default**: Network monitoring disabled by default
- **Content Redaction**: Configurable redaction for sensitive data
- **Privacy Mode**: Minimal data collection mode
- **Local Processing**: All processing happens locally
- **Anonymization**: Optional hostname and process name anonymization

## Performance Impact

The SDK is designed to be lightweight:
- **Minimal Overhead**: < 1% CPU overhead in typical usage
- **Background Collection**: Non-blocking metrics collection
- **Configurable Intervals**: Adjust collection frequency as needed
- **Efficient Instrumentation**: Uses OpenTelemetry's efficient wrapping

## Compatibility

- **Python**: 3.9+
- **Frameworks**: Compatible with any Python application
- **LLM Libraries**: Works with official SDK clients
- **Observability**: Integrates with any OpenTelemetry-compatible backend

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Issues**: Some system metrics may require elevated permissions
3. **High Resource Usage**: Adjust collection intervals if needed
4. **Missing Metrics**: Check that monitoring is enabled for specific components

### Debug Mode
```python
import logging
from MYSDK.comp_tracker.config import logger

logger.setLevel(logging.DEBUG)
```

## License

MIT License - see LICENSE file for details.

## Support

For issues, feature requests, or questions:
- Create an issue in the repository
- Check the examples and documentation
- Review the integration example for complete usage patterns

---

**AgentBay SDK** - Complete observability for AI agents and systems. 🌊
