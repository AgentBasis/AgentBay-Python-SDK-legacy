# AI Agent Tracking SDK

## Installation

```bash
pip install .
```

## Configuration

The SDK requires an API key for authentication. You can provide it in two ways:

1. Environment Variable (Recommended):
```bash
export SDK_API_KEY="your_api_key_here"
export SDK_CLIENT_ID="your_client_id_here"  # Optional
```

2. Programmatic Configuration:
```python
from tracker import AgentPerformanceTracker, APIConfig

config = APIConfig(
    api_key="your_api_key_here",      # Do NOT hardcode in production!
    client_id="your_client_id_here"    # Optional
)
tracker = AgentPerformanceTracker(config=config)
```

⚠️ **Security Warning**: Never commit API keys or sensitive credentials to source control. Always use environment variables or secure configuration management in production.

## Usage

```python
from tracker import AgentPerformanceTracker

# Initialize tracker (will use SDK_API_KEY from environment)
tracker = AgentPerformanceTracker()

# Start tracking a conversation
session_id = tracker.start_conversation("agent_123")

# Track messages
tracker.log_user_message(session_id, "Hello!")
tracker.log_agent_message(
    session_id=session_id,
    content="Hi there!",
    response_time_ms=150,
    tokens_used=10
)

# End conversation
tracker.end_conversation(session_id, quality_score=4.5)
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