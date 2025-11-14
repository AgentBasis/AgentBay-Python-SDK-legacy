# AgentBay Examples

Initialize tracing (console exporter):

```python
from agentbay.llm_tracking.otel import init_tracing
init_tracing(exporter="console")
```

Privacy (redact or disable content):

```python
from agentbay import configure_llm_privacy
configure_llm_privacy(capture_content=False)
# or
configure_llm_privacy(redactor=lambda s: s.replace("secret", "[REDACTED]"))
```

OpenAI instrumentation:

```python
from agentbay import instrument_openai
instrument_openai()

from openai import OpenAI
client = OpenAI()
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":"Hi"}])
```

Anthropic instrumentation:

```python
from agentbay import instrument_anthropic
instrument_anthropic()

from anthropic import Anthropic
client = Anthropic()
client.messages.create(model="claude-3-5-sonnet", messages=[{"role":"user","content":"Hi"}])
```

Gemini instrumentation:

```python
from agentbay import instrument_gemini
instrument_gemini()

import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-pro")
model.generate_content("Hello")
```

IBM watsonx.ai instrumentation:

```python
from agentbay import instrument_watsonx
instrument_watsonx()

from ibm_watsonx_ai.foundation_models.inference import ModelInference
mi = ModelInference(model_id="ibm/granite-13b-chat-v2")
mi.chat(messages=[{"role":"user","content":"Hi"}])
```

Grok (xAI) instrumentation (auto-detects when using xAI API):

```python
from agentbay import instrument_grok
instrument_grok()

from openai import OpenAI
# Auto-detected as Grok due to api.x.ai base_url
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key="your-xai-api-key"
)
response = client.chat.completions.create(
    model="grok-beta",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Grok with manual override:

```python
import os
# Force Grok detection even with custom proxy
os.environ["LLM_PROVIDER"] = "grok"
from agentbay import instrument_grok
instrument_grok()

# Now any OpenAI client will be treated as Grok
from openai import OpenAI
client = OpenAI(base_url="https://my-proxy.com/v1")
```

Manual custom LLM tracking:

```python
from agentbay.llm_tracking.manual import start_llm_span
span = start_llm_span(system="custom", model="my-llm")
span.add_prompt(role="user", content="Hello")
span.add_completion(content="Hi there!")
span.set_usage(prompt_tokens=5, completion_tokens=10)
span.end(finish_reason="stop")
```

