# bay_frameworks (monitoring-only)

`bay_frameworks` provides auto-instrumentation for agentic frameworks using OpenTelemetry. It includes no sessions/business logic; only telemetry.

## Install runtime deps

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp wrapt packaging
```

Frameworks are optional and only instrumented if installed: `langgraph`, `crewai`, `ag2`, `agno`, `smolagents`, `xpander-sdk`, `haystack-ai`, `agents` (OpenAI Agents), `google.adk`.

## Quick start

```python
import agentbay.bay_frameworks.instrumentation as bay

bay.instrument_all()  # enable import hook

# Import your frameworks after enabling the hook
from langgraph.graph import StateGraph
# ... your code ...
```

## Skipped namespaces

The import hook skips `agentbay.llm_tracking` and provider SDKs like `openai`, `anthropic`, `google.generativeai`, `ibm_watsonx_ai`, etc., to avoid collisions.

## Uninstrument

```python
bay.uninstrument_all()
```
