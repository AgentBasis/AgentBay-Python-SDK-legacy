"""Anthropic instrumentation

Tracks (read-only):
- Model + response IDs
- Conversation: gen_ai.prompt.{i}.role/content; gen_ai.completion.{i}.role/content/finish_reason
- Tools: request tool defs; response tool calls (when present)
- Usage: input/output/total tokens mapped to gen_ai.usage.prompt/completion/total
- Streaming: gen_ai.streaming.time_to_first_token, gen_ai.streaming.time_to_generate, gen_ai.streaming.chunk_count

Does NOT record control parameters (temperature, top_p, max_tokens, penalties).

Usage:
    from agentbay import instrument_anthropic
    instrument_anthropic()
"""

from __future__ import annotations

import json
import time
import inspect
from typing import Any, Dict, Iterable, Optional, Tuple

from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry.instrumentation.utils import unwrap as otel_unwrap

from .config import maybe_redact
from .metrics import (
    record_token_usage,
    record_streaming_metrics,
    record_message_count,
    record_tool_call_count,
    record_tool_call_event,
    record_message_event,
)


TRACER = get_tracer("llmtracker.anthropic")


class Attr:
    # System
    SYSTEM = "gen_ai.system"
    # Request
    REQ_MODEL = "gen_ai.request.model"
    REQ_HEADERS = "gen_ai.request.headers"
    REQ_STREAMING = "gen_ai.request.streaming"
    REQ_FUNCTIONS = "gen_ai.request.tools"
    # Prompts/Completions
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"
    # Response
    RESP_ID = "gen_ai.response.id"
    RESP_MODEL = "gen_ai.response.model"
    RESP_FINISH = "gen_ai.response.finish_reason"
    RESP_STOP = "gen_ai.response.stop_reason"
    # Usage
    USE_PROMPT = "gen_ai.usage.prompt_tokens"
    USE_COMP = "gen_ai.usage.completion_tokens"
    USE_TOTAL = "gen_ai.usage.total_tokens"
    # Streaming
    STREAM_TTFB = "gen_ai.streaming.time_to_first_token"
    STREAM_TTG = "gen_ai.streaming.time_to_generate"
    STREAM_CHUNKS = "gen_ai.streaming.chunk_count"


def _safe_set(span, key: str, value: Any, max_len: int = 2000):
    if value is None:
        return
    try:
        if isinstance(value, str) and len(value) > max_len:
            value = value[: max_len - 3] + "..."
        span.set_attribute(key, value)
    except Exception:
        pass


def _model_as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # pydantic
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {}


def _extract_text_from_content(content: Any) -> str:
    """Extract a simple text summary from Anthropic message content structures."""
    # content can be str or list of blocks {type: 'text'|'tool_result'|'image'...}
    if isinstance(content, str):
        return content
    txt_parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            try:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        txt_parts.append(str(item.get("text", "")))
                    elif item.get("type") == "tool_result" and "content" in item:
                        txt_parts.append(f"[Tool Result: {str(item.get('content'))}]")
                else:
                    d = _model_as_dict(item)
                    if d.get("type") == "text" and "text" in d:
                        txt_parts.append(str(d.get("text", "")))
            except Exception:
                continue
    return " ".join([p for p in txt_parts if p])


def _set_prompts_from_kwargs(span, kwargs: Dict[str, Any]):
    # System prompt
    system = kwargs.get("system")
    if system:
        _safe_set(span, f"{Attr.PROMPT}.0.role", "system")
        red = maybe_redact(system)
        if red is not None:
            _safe_set(span, f"{Attr.PROMPT}.0.content", red)
    # Messages
    messages = kwargs.get("messages", [])
    try:
        if isinstance(messages, list):
            _safe_set(span, "gen_ai.prompt.count", len(messages))
            record_message_count(len(messages))
    except Exception:
        pass
    for i, msg in enumerate(messages):
        try:
            role = None
            content = None
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                d = _model_as_dict(msg)
                role = d.get("role", "user")
                content = d.get("content", "")
            try:
                record_message_event(role=role or "user", provider="Anthropic")
            except Exception:
                pass
            content_text = _extract_text_from_content(content)
            content_text = maybe_redact(content_text)
            _safe_set(span, f"{Attr.PROMPT}.{i}.role", role)
            if content_text is not None:
                _safe_set(span, f"{Attr.PROMPT}.{i}.content", content_text)
        except Exception:
            continue


def _set_tool_definitions(span, tools: Any):
    try:
        for i, tool in enumerate(tools or []):
            d = tool if isinstance(tool, dict) else _model_as_dict(tool)
            name = d.get("name")
            desc = d.get("description")
            params = d.get("input_schema") or d.get("parameters")
            prefix = f"{Attr.REQ_FUNCTIONS}.{i}"
            _safe_set(span, f"{prefix}.name", name)
            _safe_set(span, f"{prefix}.description", desc)
            if params is not None:
                _safe_set(span, f"{prefix}.parameters", json.dumps(params))
    except Exception:
        pass


def _set_usage_from_obj(span, usage_obj: Any):
    if not usage_obj:
        return
    d = _model_as_dict(usage_obj)
    # Anthropic uses input_tokens/output_tokens
    in_tokens = d.get("input_tokens")
    out_tokens = d.get("output_tokens")
    if in_tokens is not None:
        _safe_set(span, Attr.USE_PROMPT, in_tokens)
    if out_tokens is not None:
        _safe_set(span, Attr.USE_COMP, out_tokens)
    if in_tokens is not None and out_tokens is not None:
        _safe_set(span, Attr.USE_TOTAL, in_tokens + out_tokens)
    try:
        record_token_usage(in_tokens if isinstance(in_tokens, int) else None, out_tokens if isinstance(out_tokens, int) else None)
    except Exception:
        pass


def _set_completion_from_response(span, response: Any):
    d = _model_as_dict(response)
    # IDs/models
    if "id" in d:
        _safe_set(span, Attr.RESP_ID, d.get("id"))
    if "model" in d:
        _safe_set(span, Attr.RESP_MODEL, d.get("model"))
    # Stop/final reasons
    stop_reason = d.get("stop_reason")
    if stop_reason is not None:
        _safe_set(span, Attr.RESP_STOP, stop_reason)
        _safe_set(span, Attr.RESP_FINISH, stop_reason)
    # Usage
    if "usage" in d:
        _set_usage_from_obj(span, d.get("usage"))
    # Content -> first completion index 0
    content = d.get("content")
    if content is not None:
        text = _extract_text_from_content(content)
        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
        red = maybe_redact(text)
        if red is not None:
            _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
        # Tool-use blocks
        try:
            tool_idx = 0
            if isinstance(content, list):
                for item in content:
                    block = item if isinstance(item, dict) else _model_as_dict(item)
                    if block.get("type") == "tool_use":
                        call_id = block.get("id")
                        name = block.get("name")
                        arguments = block.get("input")  # Anthropic uses `input` for arguments
                        prefix = f"{Attr.COMPLETION}.0.tool_calls.{tool_idx}"
                        _safe_set(span, f"{prefix}.id", call_id)
                        _safe_set(span, f"{prefix}.type", "function")
                        _safe_set(span, f"{prefix}.name", name)
                        if arguments is not None:
                            _safe_set(span, f"{prefix}.arguments", json.dumps(arguments))
                        try:
                            record_tool_call_event(tool_name=name, provider="Anthropic")
                        except Exception:
                            pass
                        # Child tool span
                        with TRACER.start_as_current_span(
                            name=f"tool_call.{name or 'function'}", kind=SpanKind.INTERNAL
                        ) as tool_span:
                            _safe_set(tool_span, Attr.SYSTEM, "Anthropic")
                            _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.name", name)
                            if arguments is not None:
                                _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.arguments", json.dumps(arguments))
                        tool_idx += 1
            # Record tool call count for this response
            if tool_idx > 0:
                _safe_set(span, "gen_ai.completion.tool_call_count", tool_idx)
                record_tool_call_count(tool_idx)
        except Exception:
            pass
    # Legacy `completion` field (old API)
    if "completion" in d and d.get("completion") is not None:
        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
        red = maybe_redact(d.get("completion"))
        if red is not None:
            _safe_set(span, f"{Attr.COMPLETION}.0.content", red)


def _wrap_method(trace_name: str, is_async: bool = False):
    def sync_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "Anthropic"}
        model = kwargs.get("model") if isinstance(kwargs, dict) else None
        headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
        stream = bool((kwargs or {}).get("stream", False))
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, stream)
            if headers is not None:
                _safe_set(span, Attr.REQ_HEADERS, str(headers))
            # Prompts + tool defs
            if isinstance(kwargs, dict):
                _set_prompts_from_kwargs(span, kwargs)
                if "tools" in kwargs:
                    _set_tool_definitions(span, kwargs.get("tools"))
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            # Streaming
            if stream and _is_stream_like(result):
                return _wrap_stream_result(span, result)

            # Non-stream response extraction
            try:
                _set_completion_from_response(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    async def async_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "Anthropic"}
        model = kwargs.get("model") if isinstance(kwargs, dict) else None
        headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
        stream = bool((kwargs or {}).get("stream", False))
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, stream)
            if headers is not None:
                _safe_set(span, Attr.REQ_HEADERS, str(headers))
            if isinstance(kwargs, dict):
                _set_prompts_from_kwargs(span, kwargs)
                if "tools" in kwargs:
                    _set_tool_definitions(span, kwargs.get("tools"))
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            if stream and _is_async_stream_like(result):
                return _wrap_async_stream_result(span, result)

            try:
                _set_completion_from_response(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    return async_wrapper if is_async else sync_wrapper


def _is_stream_like(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and not isinstance(obj, (dict, list, str, bytes))


def _is_async_stream_like(obj: Any) -> bool:
    return inspect.isasyncgen(obj) or hasattr(obj, "__aiter__")


def _wrap_stream_result(span, stream_obj: Iterable[Any]):
    start_time = time.time()
    first_token_time: Optional[float] = None
    chunk_count = 0

    def generator():
        nonlocal first_token_time, chunk_count
        try:
            for event in stream_obj:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                # Try to capture small bits of metadata from events
                _maybe_set_from_event(span, event)
                yield event
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if first_token_time is not None:
                total_time = time.time() - start_time
                _safe_set(span, Attr.STREAM_TTG, max(total_time - first_token_time, 0.0))
            _safe_set(span, Attr.STREAM_CHUNKS, chunk_count)
            try:
                record_streaming_metrics(ttfb=first_token_time, generate_time=(max(total_time - first_token_time, 0.0) if first_token_time is not None else None), chunks=chunk_count)
            except Exception:
                pass
            span.set_status(Status(StatusCode.OK))
            span.end()

    return generator()


def _wrap_async_stream_result(span, async_stream_obj: Any):
    start_time = time.time()
    first_token_time: Optional[float] = None
    chunk_count = 0

    async def agen():
        nonlocal first_token_time, chunk_count
        try:
            async for event in async_stream_obj:
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                _maybe_set_from_event(span, event)
                yield event
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if first_token_time is not None:
                total_time = time.time() - start_time
                _safe_set(span, Attr.STREAM_TTG, max(total_time - first_token_time, 0.0))
            _safe_set(span, Attr.STREAM_CHUNKS, chunk_count)
            try:
                record_streaming_metrics(ttfb=first_token_time, generate_time=(max(total_time - first_token_time, 0.0) if first_token_time is not None else None), chunks=chunk_count)
            except Exception:
                pass
            span.set_status(Status(StatusCode.OK))
            span.end()

    return agen()


def _maybe_set_from_event(span, event: Any):
    """Best-effort extraction from Anthropic streaming events without strict types."""
    try:
        d = _model_as_dict(event)
        # Message id/model may appear on event.message
        msg = d.get("message") or d
        if isinstance(msg, dict):
            if "id" in msg:
                _safe_set(span, Attr.RESP_ID, msg.get("id"))
            if "model" in msg:
                _safe_set(span, Attr.RESP_MODEL, msg.get("model"))
            if "usage" in msg:
                _set_usage_from_obj(span, msg.get("usage"))
            if "stop_reason" in msg:
                _safe_set(span, Attr.RESP_STOP, msg.get("stop_reason"))
                _safe_set(span, Attr.RESP_FINISH, msg.get("stop_reason"))
    except Exception:
        pass


def instrument_anthropic() -> None:
    """Enable automatic tracking for Anthropic API calls."""
    # Messages.create (modern)
    wrap_function_wrapper(
        "anthropic.resources.messages", "Messages.create", _wrap_method("anthropic.messages.create")
    )
    wrap_function_wrapper(
        "anthropic.resources.messages", "AsyncMessages.create", _wrap_method("anthropic.messages.create", is_async=True)
    )

    # Legacy Completions.create
    try:
        wrap_function_wrapper(
            "anthropic.resources.completions", "Completions.create", _wrap_method("anthropic.completions.create")
        )
        wrap_function_wrapper(
            "anthropic.resources.completions",
            "AsyncCompletions.create",
            _wrap_method("anthropic.completions.create", is_async=True),
        )
    except Exception:
        pass

    # Streaming (messages.stream)
    try:
        wrap_function_wrapper(
            "anthropic.resources.messages.messages", "Messages.stream", _wrap_method("anthropic.messages.stream")
        )
        wrap_function_wrapper(
            "anthropic.resources.messages.messages",
            "AsyncMessages.stream",
            _wrap_method("anthropic.messages.stream", is_async=True),
        )
    except Exception:
        pass


def uninstrument_anthropic() -> None:
    """Disable Anthropic tracking."""
    targets = [
        ("anthropic.resources.messages", "Messages.create"),
        ("anthropic.resources.messages", "AsyncMessages.create"),
        ("anthropic.resources.completions", "Completions.create"),
        ("anthropic.resources.completions", "AsyncCompletions.create"),
        ("anthropic.resources.messages.messages", "Messages.stream"),
        ("anthropic.resources.messages.messages", "AsyncMessages.stream"),
    ]
    for mod, meth in targets:
        try:
            otel_unwrap(mod, meth)
        except Exception:
            pass
