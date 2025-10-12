# Python SDK

## Installation

```bash
pip install .
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

## Usage

The easiest way to get started is with `quick_setup`. This function initializes both LLM and system tracking with sensible defaults.

```python
from agentbay import quick_setup
import openai # or any other supported library

# Ensure OPENAI_API_KEY is set in your environment
# pip install openai

# Initialize both LLM and system tracking
quick_setup(
    llm_providers=["openai"],
    system_monitoring=True,
    collection_interval=30,
    exporter="console" # Prints telemetry to the console
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

## Features

- Agent operations tracking
- Performance metrics
- Message-level tracking
- Conversation history
- Security features
- Compliance tracking
- LLM usage tracking

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
