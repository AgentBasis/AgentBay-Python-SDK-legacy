
"""LLM tracking SDK 

Exports:
- instrument_openai(): attach wrappers to OpenAI SDK (v1/v0) to passively track
  conversations, tool calls, token usage, and streaming timings using neutral
  `gen_ai.*` attributes. No control parameters are recorded.
- uninstrument_openai(): best-effort unwrapping (no-op if unavailable).
- instrument_grok(): attach wrappers to OpenAI SDK for Grok (xAI) to passively track
  conversations, tool calls, token usage, and streaming timings using neutral
  `gen_ai.*` attributes. Auto-detects when base_url contains 'api.x.ai'.
- uninstrument_grok(): best-effort unwrapping (no-op if unavailable).
"""

from .openai import instrument_openai, uninstrument_openai
from .anthropic import instrument_anthropic, uninstrument_anthropic
from .gemini import instrument_gemini, uninstrument_gemini
from .watsonx import instrument_watsonx, uninstrument_watsonx
from .grok import instrument_grok, uninstrument_grok
from .config import configure as configure_privacy

__all__ = [
    "instrument_openai",
    "uninstrument_openai",
    "instrument_anthropic",
    "uninstrument_anthropic",
    "instrument_gemini",
    "uninstrument_gemini",
    "instrument_watsonx",
    "uninstrument_watsonx",
    "instrument_grok",
    "uninstrument_grok",
    "configure_privacy",
]
