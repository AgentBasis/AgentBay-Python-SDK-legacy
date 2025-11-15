"""Manual API for tracking custom/unsupported LLMs.

Example:
    from agentbay.llm_tracking.manual import start_llm_span
    span = start_llm_span(system="custom", model="my-llm")
    span.add_prompt(role="user", content="Hello")
    # call your LLM...
    span.add_completion(role="assistant", content="Hi!")
    span.set_usage(prompt_tokens=10, completion_tokens=12)
    span.end(finish_reason="stop")
"""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode

from .config import maybe_redact


TRACER = get_tracer("llmtracker.manual")


class Attr:
    SYSTEM = "gen_ai.system"
    REQ_MODEL = "gen_ai.request.model"
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"
    USE_PROMPT = "gen_ai.usage.prompt_tokens"
    USE_COMP = "gen_ai.usage.completion_tokens"
    USE_TOTAL = "gen_ai.usage.total_tokens"
    STREAM_TTFB = "gen_ai.streaming.time_to_first_token"
    STREAM_TTG = "gen_ai.streaming.time_to_generate"
    STREAM_CHUNKS = "gen_ai.streaming.chunk_count"


def _safe_set(span, key: str, value: Any):
    if value is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


class LLMSpan:
    """Manual span for tracking custom or unsupported LLM calls."""
    
    def __init__(self, system: str, model: Optional[str]):
        self._span_cm = TRACER.start_as_current_span("custom.llm", kind=SpanKind.CLIENT)
        self._span = self._span_cm.__enter__()
        _safe_set(self._span, Attr.SYSTEM, system)
        if model:
            _safe_set(self._span, Attr.REQ_MODEL, model)
        self._stream_started_at = None
        self._first_token_at = None
        self._chunks = 0

    def add_prompt(self, *, role: str = "user", content: Optional[str] = None, index: int = 0):
        _safe_set(self._span, f"{Attr.PROMPT}.{index}.role", role)
        red = maybe_redact(content) if content is not None else None
        if red is not None:
            _safe_set(self._span, f"{Attr.PROMPT}.{index}.content", red)
        return self

    def add_completion(self, *, role: str = "assistant", content: Optional[str] = None, index: int = 0, finish_reason: Optional[str] = None):
        _safe_set(self._span, f"{Attr.COMPLETION}.{index}.role", role)
        red = maybe_redact(content) if content is not None else None
        if red is not None:
            _safe_set(self._span, f"{Attr.COMPLETION}.{index}.content", red)
        if finish_reason is not None:
            _safe_set(self._span, f"{Attr.COMPLETION}.{index}.finish_reason", finish_reason)
        return self

    def add_tool_call(self, *, index: int, name: str, arguments: Any = None, tool_index: int = 0):
        prefix = f"{Attr.COMPLETION}.{index}.tool_calls.{tool_index}"
        _safe_set(self._span, f"{prefix}.type", "function")
        _safe_set(self._span, f"{prefix}.name", name)
        if arguments is not None:
            _safe_set(self._span, f"{prefix}.arguments", str(arguments))
        return self

    def set_usage(self, *, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None):
        if prompt_tokens is not None:
            _safe_set(self._span, Attr.USE_PROMPT, prompt_tokens)
        if completion_tokens is not None:
            _safe_set(self._span, Attr.USE_COMP, completion_tokens)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens is not None:
            _safe_set(self._span, Attr.USE_TOTAL, total_tokens)
        return self

    def stream_start(self):
        import time

        self._stream_started_at = time.time()
        self._first_token_at = None
        self._chunks = 0
        return self

    def stream_chunk(self):
        import time

        self._chunks += 1
        if self._first_token_at is None and self._stream_started_at is not None:
            self._first_token_at = time.time()
            _safe_set(self._span, Attr.STREAM_TTFB, self._first_token_at - self._stream_started_at)
        return self

    def stream_end(self):
        import time

        if self._first_token_at is not None and self._stream_started_at is not None:
            _safe_set(self._span, Attr.STREAM_TTG, time.time() - self._first_token_at)
        _safe_set(self._span, Attr.STREAM_CHUNKS, self._chunks)
        return self

    def end(self, *, finish_reason: Optional[str] = None, ok: bool = True):
        if finish_reason is not None:
            _safe_set(self._span, f"{Attr.COMPLETION}.0.finish_reason", finish_reason)
        if ok:
            self._span.set_status(Status(StatusCode.OK))
        self._span_cm.__exit__(None, None, None)


def start_llm_span(*, system: str = "custom", model: Optional[str] = None) -> LLMSpan:
    """
    Create a manual LLM span for custom tracking.
    
    Args:
        system: System identifier for the LLM (e.g., "custom", "my-llm")
        model: Optional model name/identifier
    
    Returns:
        LLMSpan instance for tracking LLM interactions
    """
    return LLMSpan(system=system, model=model)

