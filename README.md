# AgentBay Python SDK
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AgentBay-AI/AgentBay-Python-SDK)

Complete AI Agent Monitoring Solution - Track LLM usage, system resources, agentic frameworks, and security events.

## Installation

### From Source (Development)

```bash
pip install .
```

### From PyPI (Once Published)

```bash
pip install agentbay
```

## Configuration

The SDK uses OpenTelemetry for exporting data. Configuration primarily involves setting up an OTel exporter and tuning the data collection settings.

### Telemetry Exporter

By default, `quick_setup` uses a console exporter, which is useful for debugging. For production, you'll want to configure an OTLP exporter to send data to an observability backend like AgentBay, Jaeger, or Honeycomb.

```python
from agentbay import init_tracing

# Point to your OTLP collector endpoint
init_tracing(exporter="otlp-http", endpoint="http://your-collector-endpoint:4318/v1/traces")
```

### Data Collection

You can configure LLM and system tracking separately.

**LLM Tracking Privacy:**

Control what data is captured from LLM interactions.

```python
from agentbay import configure_llm_privacy

# Disable capturing prompt/completion content entirely
configure_llm_privacy(capture_content=False)

# Or, provide a custom redactor function
def simple_redactor(text: str) -> str:
    return "[REDACTED]"

configure_llm_privacy(redactor=simple_redactor)
```

**System Tracking:**

Tune which system components are monitored and their collection intervals.

```python
from agentbay import configure_system

configure_system(
    enable_network_monitoring=True,  # Default is False
    cpu_alert_threshold=75.0,        # Alert when CPU > 75%
    default_collection_interval=60.0 # Collect metrics every 60 seconds
)
```

You can also configure system tracking via environment variables (e.g., `AGENTBAY_ENABLE_NETWORK_MONITORING=true`, `AGENTBAY_COLLECTION_INTERVAL=60`).

### Individual Provider Instrumentation

You can also instrument LLM providers individually:

```python
from agentbay import instrument_openai, instrument_anthropic, instrument_gemini

# Instrument specific providers
instrument_openai()
instrument_anthropic()
instrument_gemini()

# Use your clients as normal - they're automatically tracked
from openai import OpenAI
from anthropic import Anthropic
from google import generativeai as genai

# All calls will be automatically tracked
```

## Usage

### Quick Start

The easiest way to get started is with `quick_setup`. This function initializes LLM tracking, system monitoring, framework instrumentation, and security monitoring with sensible defaults.

```python
from agentbay import quick_setup
import openai  # or any other supported library

# Ensure OPENAI_API_KEY is set in your environment
# pip install openai

# Initialize comprehensive monitoring
result = quick_setup(
    llm_providers=["openai"],
    system_monitoring=True,
    enable_framework_instrumentation=True,  # Auto-instrument langgraph, crewai, etc.
    enable_security_monitoring=True,        # Enable security monitoring
    collection_interval=30,
    exporter="console"  # Prints telemetry to the console
)

# Now, use your LLM client as usual.
# The SDK will automatically track usage.
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke about observability."}]
)
print(response.choices[0].message.content)
```

### Agentic Framework Instrumentation

Auto-instrument popular agentic frameworks:

```python
from agentbay import instrument_all, get_active_libraries

# Enable auto-instrumentation for all supported frameworks
instrument_all()

# Check which frameworks are detected
active = get_active_libraries()
print(f"Detected frameworks: {active}")

# Now import your frameworks - they'll be automatically instrumented
from langgraph.graph import StateGraph
from crewai import Agent, Task, Crew
# ... your agentic code ...
```

Supported frameworks: `langgraph`, `crewai`, `ag2`, `agno`, `smolagents`, `xpander-sdk`, `haystack-ai`, `agents` (OpenAI Agents), `google.adk`.

### Security Monitoring

Monitor agent interactions for security threats:

```python
from agentbay import SecurityMonitor, quick_setup_security

# Quick setup with defaults
monitor = quick_setup_security(
    enable_context_monitoring=True,
    enable_compliance_tracking=True,
    risk_threshold=0.7
)

# Track agent interactions
monitor.track_interaction(
    agent_id="agent_1",
    user_input="Hello, can you help me?",
    agent_response="Of course! How can I assist you?",
    session_id="session_123"
)
```

Security features include: prompt injection detection, jailbreak detection, social engineering recognition, compliance tracking (GDPR, CCPA, HIPAA), and risk assessment.

## Features

### LLM Tracking
- **Multi-Provider Support**: OpenAI, Anthropic, Gemini, Grok, Watson X
- **Conversation Tracking**: Full message history with privacy controls
- **Token Usage**: Track prompt, completion, and total tokens
- **Streaming Metrics**: Time-to-first-token, generation time, chunk counts
- **Tool Call Monitoring**: Track function calls and responses

### System Component Tracking
- **CPU Monitoring**: Usage, cores, frequency
- **Memory Tracking**: RAM usage, availability, swap
- **Disk Monitoring**: Usage across all partitions
- **Network I/O**: Bandwidth, connections (optional)
- **Process Tracking**: Resource consumption per process
- **Health Indicators**: Alert thresholds and status checks

### Agentic Framework Instrumentation
- **Auto-Detection**: Automatically detects and instruments frameworks
- **Supported Frameworks**: LangGraph, CrewAI, AG2, Agno, SmolAgents, Xpander, Haystack, OpenAI Agents, Google ADK
- **Zero Configuration**: Works out of the box with import hooks
- **OpenTelemetry Integration**: Full trace and metric support

### Security Monitoring
- **Threat Detection**: Prompt injection, jailbreak attempts, social engineering
- **Compliance Tracking**: GDPR, CCPA, HIPAA, SOC2, ISO27001, PCI-DSS
- **Risk Assessment**: Multi-factor risk scoring
- **Audit Logging**: Immutable audit trails
- **Privacy Controls**: Configurable data minimization

## Best Practices

1. API Key Security:
   - Use environment variables for API keys
   - Never hardcode sensitive data
   - Rotate API keys periodically

2. Rate Limiting:
   - Respects 5000 requests/minute limit
   - Automatic retry with exponential backoff
   - Handles 429 responses gracefully

3. Data Validation:
   - Agent status must be: 'active', 'idle', 'busy', or 'error'
   - Quality scores: 1.0 to 5.0
   - Response times in milliseconds (2 decimal places)
   - All timestamps in milliseconds
   - JSON-serializable metadata

4. Error Handling:
   - Comprehensive error handling
   - Automatic retries for transient failures
   - Offline event queuing
   - Graceful degradation

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check types
mypy .
``` 
