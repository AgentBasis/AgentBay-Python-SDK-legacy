"""Optional metrics helpers using OpenTelemetry metrics API.

If the metrics SDK/provider isn’t configured, calls will no-op.
"""

from __future__ import annotations

from typing import Optional

try:
    from opentelemetry.metrics import get_meter
except Exception:  # pragma: no cover
    get_meter = None  # type: ignore


_initialized = False
_token_counter = None
_ttfb_hist = None
_gen_hist = None
_chunk_counter = None
_choice_counter = None
_embed_size_hist = None
_embed_exc_counter = None
_image_exc_counter = None
_message_counter = None
_tool_call_counter = None
_tool_call_event_counter = None
_message_event_counter = None


def _init():
    global _initialized, _token_counter, _ttfb_hist, _gen_hist, _chunk_counter
    global _choice_counter, _embed_size_hist, _embed_exc_counter, _image_exc_counter
    global _message_counter, _tool_call_counter, _tool_call_event_counter, _message_event_counter
    if _initialized:
        return
    if get_meter is None:
        _initialized = True
        return
    try:
        meter = get_meter("agentbay.llmtracker", version="0.1.0")
        _token_counter = meter.create_counter("gen_ai.client.token.usage", unit="token")
        _ttfb_hist = meter.create_histogram("gen_ai.streaming.time_to_first_token", unit="s")
        _gen_hist = meter.create_histogram("gen_ai.streaming.time_to_generate", unit="s")
        _chunk_counter = meter.create_counter("gen_ai.streaming.chunks", unit="chunk")
        _choice_counter = meter.create_counter("gen_ai.client.choices", unit="choice")
        _embed_size_hist = meter.create_histogram("gen_ai.embeddings.vector_size", unit="element")
        _embed_exc_counter = meter.create_counter("gen_ai.embeddings.exceptions", unit="error")
        _image_exc_counter = meter.create_counter("gen_ai.images.exceptions", unit="error")
        _message_counter = meter.create_counter("gen_ai.client.messages", unit="message")
        _tool_call_counter = meter.create_counter("gen_ai.client.tool_calls", unit="tool")
        _tool_call_event_counter = meter.create_counter("gen_ai.client.tool_call.events", unit="event")
        _message_event_counter = meter.create_counter("gen_ai.client.message.events", unit="event")
    except Exception:
        pass
    _initialized = True


def record_token_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]):
    _init()
    try:
        total = 0
        if isinstance(prompt_tokens, int):
            total += prompt_tokens
        if isinstance(completion_tokens, int):
            total += completion_tokens
        if _token_counter and total:
            _token_counter.add(total)
    except Exception:
        pass


def record_streaming_metrics(*, ttfb: Optional[float], generate_time: Optional[float], chunks: Optional[int]):
    _init()
    try:
        if _ttfb_hist and isinstance(ttfb, (int, float)):
            _ttfb_hist.record(float(ttfb))
        if _gen_hist and isinstance(generate_time, (int, float)):
            _gen_hist.record(float(generate_time))
        if _chunk_counter and isinstance(chunks, int) and chunks > 0:
            _chunk_counter.add(chunks)
    except Exception:
        pass


def record_choice_count(count: Optional[int]):
    _init()
    try:
        if _choice_counter and isinstance(count, int) and count > 0:
            _choice_counter.add(count)
    except Exception:
        pass


def record_embedding_vector_size(size: Optional[int]):
    _init()
    try:
        if _embed_size_hist and isinstance(size, int) and size > 0:
            _embed_size_hist.record(size)
    except Exception:
        pass


def record_operation_exception(kind: str):
    _init()
    try:
        if kind == "embeddings" and _embed_exc_counter:
            _embed_exc_counter.add(1)
        elif kind == "images" and _image_exc_counter:
            _image_exc_counter.add(1)
    except Exception:
        pass


def record_message_count(count: Optional[int]):
    _init()
    try:
        if _message_counter and isinstance(count, int) and count > 0:
            _message_counter.add(count)
    except Exception:
        pass


def record_tool_call_count(count: Optional[int]):
    _init()
    try:
        if _tool_call_counter and isinstance(count, int) and count > 0:
            _tool_call_counter.add(count)
    except Exception:
        pass


def record_tool_call_event(*, tool_name: Optional[str] = None, provider: Optional[str] = None):
    _init()
    try:
        if _tool_call_event_counter:
            attrs = {}
            if isinstance(tool_name, str) and tool_name:
                attrs["tool_name"] = tool_name
            if isinstance(provider, str) and provider:
                attrs["provider"] = provider
            # Some SDKs accept attributes kwarg; fall back gracefully if unsupported
            try:
                _tool_call_event_counter.add(1, attributes=attrs)  # type: ignore
            except Exception:
                _tool_call_event_counter.add(1)
    except Exception:
        pass


def record_message_event(*, role: Optional[str] = None, provider: Optional[str] = None):
    _init()
    try:
        if _message_event_counter:
            attrs = {}
            if isinstance(role, str) and role:
                attrs["role"] = role
            if isinstance(provider, str) and provider:
                attrs["provider"] = provider
            try:
                _message_event_counter.add(1, attributes=attrs)  # type: ignore
            except Exception:
                _message_event_counter.add(1)
    except Exception:
        pass
