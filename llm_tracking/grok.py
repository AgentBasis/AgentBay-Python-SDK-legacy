"""Grok (xAI) instrumentation 

Tracks (read-only):
- Model + response IDs
- Conversation: gen_ai.prompt.{i}.role/content; gen_ai.completion.{i}.role/content/finish_reason
- Tools: request tool defs; response tool calls (and child tool spans)
- Usage: gen_ai.usage.prompt_tokens, gen_ai.usage.completion_tokens, gen_ai.usage.total_tokens, gen_ai.usage.reasoning_tokens
- Streaming: gen_ai.streaming.time_to_first_token, gen_ai.streaming.time_to_generate, gen_ai.streaming.chunk_count

Does NOT record control parameters (temperature, top_p, max_tokens, penalties).

Auto-detects Grok when base_url contains 'api.x.ai', with manual override support.

Usage:
    from agentbay import instrument_grok
    instrument_grok()
"""

from __future__ import annotations

import json
import time
import inspect
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Callable, AsyncIterable

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
    record_embedding_vector_size,
    record_operation_exception,
)


TRACER = get_tracer("llmtracker.grok")


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
    RESP_FINGERPRINT = "gen_ai.response.system_fingerprint"
    # Usage
    USE_PROMPT = "gen_ai.usage.prompt_tokens"
    USE_COMP = "gen_ai.usage.completion_tokens"
    USE_TOTAL = "gen_ai.usage.total_tokens"
    USE_REASON = "gen_ai.usage.reasoning_tokens"
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


def _is_grok_client(client: Any) -> bool:
    """Detect if client is configured for Grok (xAI)."""
    # Check explicit override first
    override_provider = os.getenv("LLM_PROVIDER", "").lower()
    if override_provider == "grok":
        return True
    if override_provider and override_provider != "grok":
        return False

    # Auto-detection: check base_url
    try:
        # Resource objects typically hold a reference to the OpenAI client
        c = client
        for attr in ("client", "_client"):
            if hasattr(c, attr):
                c = getattr(c, attr)
        base_url = getattr(c, "_base_url", "") or getattr(c, "base_url", "")
        if isinstance(base_url, str) and "api.x.ai" in base_url.lower():
            return True
    except Exception:
        pass

    return False


def _extract_messages(messages: Any) -> Tuple[int, Dict[str, Any]]:
    """Extract and normalize messages from OpenAI-style format."""
    if not messages:
        return 0, {}

    attrs = {}
    count = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            msg = _model_as_dict(msg)

        role = msg.get("role", "")
        content = msg.get("content", "")

        if role and content:
            # Redact content for privacy
            redacted_content = maybe_redact(content)
            attrs[f"{Attr.PROMPT}.{i}.role"] = role
            attrs[f"{Attr.PROMPT}.{i}.content"] = redacted_content
            count += 1
        try:
            # Record message event for metrics (best-effort)
            record_message_event(role=role or "user", provider="Grok")
        except Exception:
            pass

    return count, attrs


def _extract_tools(tools: Any) -> Tuple[int, Dict[str, Any]]:
    """Extract tool definitions from OpenAI-style tools array."""
    if not tools:
        return 0, {}

    attrs = {}
    count = 0

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            tool = _model_as_dict(tool)

        tool_def = tool.get("function", tool)
        if isinstance(tool_def, dict):
            name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", {})

            if name:
                attrs[f"{Attr.REQ_FUNCTIONS}.{i}.name"] = name
                if description:
                    attrs[f"{Attr.REQ_FUNCTIONS}.{i}.description"] = description
                if parameters:
                    attrs[f"{Attr.REQ_FUNCTIONS}.{i}.parameters"] = json.dumps(parameters)
                count += 1

    return count, attrs


def _extract_usage(usage: Any) -> Dict[str, Any]:
    """Extract token usage from OpenAI-style usage object."""
    if not usage:
        return {}

    if not isinstance(usage, dict):
        usage = _model_as_dict(usage)

    attrs = {}

    # Standard token counts
    if "prompt_tokens" in usage:
        attrs[Attr.USE_PROMPT] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        attrs[Attr.USE_COMP] = usage["completion_tokens"]
    if "total_tokens" in usage:
        attrs[Attr.USE_TOTAL] = usage["total_tokens"]

    # Reasoning tokens (Grok-specific or OpenAI-style)
    reasoning_tokens = (
        usage.get("reasoning_tokens") or
        usage.get("completion_tokens_details", {}).get("reasoning_tokens") or
        usage.get("output_tokens_details", {}).get("reasoning_tokens")
    )
    if reasoning_tokens:
        attrs[Attr.USE_REASON] = reasoning_tokens

    return attrs


def _extract_completions(choices: Any) -> Tuple[int, Dict[str, Any]]:
    """Extract completions from OpenAI-style choices array."""
    if not choices:
        return 0, {}

    attrs = {}
    count = 0

    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            choice = _model_as_dict(choice)

        message = choice.get("message", {})
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason")

            if content:
                # Redact content for privacy
                redacted_content = maybe_redact(content)
                attrs[f"{Attr.COMPLETION}.{i}.role"] = role
                attrs[f"{Attr.COMPLETION}.{i}.content"] = redacted_content
                if finish_reason:
                    attrs[f"{Attr.COMPLETION}.{i}.finish_reason"] = finish_reason
                count += 1

    return count, attrs


def _extract_tool_calls(message: Any, completion_index: int = 0) -> Tuple[int, Dict[str, Any]]:
    """Extract tool calls from assistant message."""
    if not isinstance(message, dict):
        message = _model_as_dict(message)

    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        return 0, {}

    attrs = {}
    count = 0

    for j, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            tool_call = _model_as_dict(tool_call)

        call_id = tool_call.get("id", "")
        call_type = tool_call.get("type", "")
        function = tool_call.get("function", {})

        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments", "")

            if name:
                prefix = f"{Attr.COMPLETION}.{completion_index}.tool_calls.{j}"
                attrs[f"{prefix}.id"] = call_id
                attrs[f"{prefix}.type"] = call_type
                attrs[f"{prefix}.name"] = name
                attrs[f"{prefix}.arguments"] = arguments
                count += 1

    return count, attrs


def _wrap_chat_completions_create(wrapped, instance, args, kwargs):
    """Wrap OpenAI-style chat completions create method for Grok detection."""
    if not _is_grok_client(instance):
        return wrapped(*args, **kwargs)

    start_time = time.time()

    # Extract request parameters
    request_model = kwargs.get("model", "")
    is_streaming = kwargs.get("stream", False)
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])

    with TRACER.start_as_current_span(
        "chat.completions.create",
        kind=SpanKind.CLIENT,
    ) as span:
        # Set system identifier
        _safe_set(span, Attr.SYSTEM, "Grok")

        # Set request attributes
        _safe_set(span, Attr.REQ_MODEL, request_model)
        _safe_set(span, Attr.REQ_STREAMING, is_streaming)
        headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
        if headers is not None:
            _safe_set(span, Attr.REQ_HEADERS, str(headers))

        # Extract and set messages
        msg_count, msg_attrs = _extract_messages(messages)
        for key, value in msg_attrs.items():
            _safe_set(span, key, value)
        record_message_count(msg_count)

        # Extract and set tools
        tool_count, tool_attrs = _extract_tools(tools)
        for key, value in tool_attrs.items():
            _safe_set(span, key, value)

        try:
            # Call the original method
            result = wrapped(*args, **kwargs)

            if is_streaming:
                # Handle streaming response
                return _wrap_stream_response(result, span, start_time)
            else:
                # Handle non-streaming response
                out = _process_non_stream_response(result, span)
                span.set_status(Status(StatusCode.OK))
                return out

        except Exception as e:
            # Optional: record_operation_exception(kind="chat")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def _wrap_stream_response(stream, span, start_time):
    """Wrap streaming response to track metrics."""
    class StreamWrapper:
        def __init__(self, stream_iter):
            self._stream = stream_iter
            self._first_chunk_time = None
            self._chunk_count = 0
            self._span = span
            self._start_time = start_time

        def __iter__(self):
            return self

        def __next__(self):
            try:
                chunk = next(self._stream)

                # Track first chunk time
                if self._first_chunk_time is None:
                    self._first_chunk_time = time.time()
                    ttfb = self._first_chunk_time - self._start_time
                    _safe_set(self._span, Attr.STREAM_TTFB, ttfb)

                self._chunk_count += 1

                # Try to extract response metadata from chunk
                self._extract_stream_metadata(chunk)

                return chunk

            except StopIteration:
                # Stream ended - record final metrics
                end_time = time.time()
                ttg = end_time - self._start_time
                _safe_set(self._span, Attr.STREAM_TTG, ttg)
                _safe_set(self._span, Attr.STREAM_CHUNKS, self._chunk_count)
                record_streaming_metrics(
                    ttfb=(self._first_chunk_time - self._start_time) if self._first_chunk_time else None,
                    generate_time=ttg,
                    chunks=self._chunk_count,
                )
                raise

        def _extract_stream_metadata(self, chunk):
            """Extract metadata from streaming chunk."""
            try:
                if hasattr(chunk, 'model_dump'):
                    chunk_data = chunk.model_dump()
                elif hasattr(chunk, '__dict__'):
                    chunk_data = dict(chunk.__dict__)
                else:
                    chunk_data = chunk if isinstance(chunk, dict) else {}

                # Extract response ID and model
                if 'id' in chunk_data and not hasattr(self._span, '_resp_id_set'):
                    _safe_set(self._span, Attr.RESP_ID, chunk_data['id'])
                    self._span._resp_id_set = True

                if 'model' in chunk_data:
                    _safe_set(self._span, Attr.RESP_MODEL, chunk_data['model'])

                # Extract usage if present
                if 'usage' in chunk_data:
                    usage_attrs = _extract_usage(chunk_data['usage'])
                    for key, value in usage_attrs.items():
                        _safe_set(self._span, key, value)
                    try:
                        pt = usage_attrs.get(Attr.USE_PROMPT)
                        ct = usage_attrs.get(Attr.USE_COMP)
                        record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
                    except Exception:
                        pass

                # Extract finish reason
                if 'choices' in chunk_data:
                    for choice in chunk_data['choices']:
                        if isinstance(choice, dict) and 'finish_reason' in choice:
                            _safe_set(self._span, Attr.RESP_FINISH, choice['finish_reason'])
                            _safe_set(self._span, Attr.RESP_STOP, choice['finish_reason'])

            except Exception:
                pass  # Don't break streaming for metadata extraction errors

    return StreamWrapper(stream)


async def _wrap_stream_response_async(stream, span, start_time):
    """Wrap async streaming response to track metrics."""
    class AsyncStreamWrapper:
        def __init__(self, stream_iter):
            self._stream = stream_iter
            self._first_chunk_time = None
            self._chunk_count = 0
            self._span = span
            self._start_time = start_time

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                chunk = await self._stream.__anext__()

                # Track first chunk time
                if self._first_chunk_time is None:
                    self._first_chunk_time = time.time()
                    ttfb = self._first_chunk_time - self._start_time
                    _safe_set(self._span, Attr.STREAM_TTFB, ttfb)

                self._chunk_count += 1

                # Try to extract response metadata from chunk
                self._extract_stream_metadata(chunk)

                return chunk

            except StopAsyncIteration:
                # Stream ended - record final metrics
                end_time = time.time()
                ttg = end_time - self._start_time
                _safe_set(self._span, Attr.STREAM_TTG, ttg)
                _safe_set(self._span, Attr.STREAM_CHUNKS, self._chunk_count)
                record_streaming_metrics(
                    ttfb=(self._first_chunk_time - self._start_time) if self._first_chunk_time else None,
                    generate_time=ttg,
                    chunks=self._chunk_count,
                )
                raise

        def _extract_stream_metadata(self, chunk):
            """Extract metadata from streaming chunk."""
            try:
                if hasattr(chunk, 'model_dump'):
                    chunk_data = chunk.model_dump()
                elif hasattr(chunk, '__dict__'):
                    chunk_data = dict(chunk.__dict__)
                else:
                    chunk_data = chunk if isinstance(chunk, dict) else {}

                # Extract response ID and model
                if 'id' in chunk_data and not hasattr(self._span, '_resp_id_set'):
                    _safe_set(self._span, Attr.RESP_ID, chunk_data['id'])
                    self._span._resp_id_set = True

                if 'model' in chunk_data:
                    _safe_set(self._span, Attr.RESP_MODEL, chunk_data['model'])

                # Extract usage if present
                if 'usage' in chunk_data:
                    usage_attrs = _extract_usage(chunk_data['usage'])
                    for key, value in usage_attrs.items():
                        _safe_set(self._span, key, value)
                    try:
                        pt = usage_attrs.get(Attr.USE_PROMPT)
                        ct = usage_attrs.get(Attr.USE_COMP)
                        record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
                    except Exception:
                        pass

                # Extract finish reason
                if 'choices' in chunk_data:
                    for choice in chunk_data['choices']:
                        if isinstance(choice, dict) and 'finish_reason' in choice:
                            _safe_set(self._span, Attr.RESP_FINISH, choice['finish_reason'])
                            _safe_set(self._span, Attr.RESP_STOP, choice['finish_reason'])

            except Exception:
                pass  # Don't break streaming for metadata extraction errors

    return AsyncStreamWrapper(stream)


def _process_non_stream_response(result, span):
    """Process non-streaming response and extract metadata."""
    try:
        if hasattr(result, 'model_dump'):
            result_data = result.model_dump()
        elif hasattr(result, '__dict__'):
            result_data = dict(result.__dict__)
        else:
            result_data = result if isinstance(result, dict) else {}

        # Extract response metadata
        _safe_set(span, Attr.RESP_ID, result_data.get('id'))
        _safe_set(span, Attr.RESP_MODEL, result_data.get('model'))

        # Extract usage
        if 'usage' in result_data:
            usage_attrs = _extract_usage(result_data['usage'])
            for key, value in usage_attrs.items():
                _safe_set(span, key, value)
            try:
                pt = usage_attrs.get(Attr.USE_PROMPT)
                ct = usage_attrs.get(Attr.USE_COMP)
                record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
            except Exception:
                pass

        # Extract completions and tool calls
        if 'choices' in result_data:
            comp_count, comp_attrs = _extract_completions(result_data['choices'])
            for key, value in comp_attrs.items():
                _safe_set(span, key, value)

            total_tool_calls = 0
            for idx, choice in enumerate(result_data['choices']):
                if isinstance(choice, dict) and 'message' in choice:
                    tool_count, tool_attrs = _extract_tool_calls(choice['message'], completion_index=idx)
                    for key, value in tool_attrs.items():
                        _safe_set(span, key, value)
                    total_tool_calls += tool_count
                    if tool_count > 0:
                        _create_tool_child_spans(span, choice['message'].get('tool_calls', []))
            if total_tool_calls > 0:
                _safe_set(span, "gen_ai.completion.tool_call_count", total_tool_calls)
                record_tool_call_count(total_tool_calls)

            # Extract finish reason
            if result_data['choices'][0].get('finish_reason'):
                finish_reason = result_data['choices'][0]['finish_reason']
                _safe_set(span, Attr.RESP_FINISH, finish_reason)
                _safe_set(span, Attr.RESP_STOP, finish_reason)

        # Extract system fingerprint if present
        if 'system_fingerprint' in result_data:
            _safe_set(span, Attr.RESP_FINGERPRINT, result_data['system_fingerprint'])

        # Fallback for Responses API: output_text content
        if 'choices' not in result_data:
            if isinstance(result_data.get('output_text'), str):
                red = maybe_redact(result_data.get('output_text'))
                if red:
                    _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
                    _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")

    except Exception as e:
        try:
            record_operation_exception(kind="chat")
        except Exception:
            pass
        span.set_status(Status(StatusCode.ERROR, f"Error processing response: {e}"))

    return result


def _create_tool_child_spans(parent_span, tool_calls):
    """Create child spans for tool calls."""
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            tool_call = _model_as_dict(tool_call)

        function = tool_call.get('function', {})
        if isinstance(function, dict):
            tool_name = function.get('name', 'unknown_tool')
            arguments = function.get('arguments', '')

            with TRACER.start_as_current_span(
                f"tool_call.{tool_name}",
                kind=SpanKind.INTERNAL,
            ) as tool_span:
                _safe_set(tool_span, Attr.SYSTEM, "Grok")
                _safe_set(tool_span, "gen_ai.tool_call.arguments", arguments)
                try:
                    record_tool_call_event(tool_name=tool_name, provider="Grok")
                except Exception:
                    pass


def _wrap_chat_completions_create_async(wrapped, instance, args, kwargs):
    """Wrap async OpenAI-style chat completions create method for Grok detection."""
    if not _is_grok_client(instance):
        return wrapped(*args, **kwargs)

    start_time = time.time()

    # Extract request parameters (same as sync version)
    request_model = kwargs.get("model", "")
    is_streaming = kwargs.get("stream", False)
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])

    async def _async_wrapper():
        with TRACER.start_as_current_span(
            "chat.completions.create",
            kind=SpanKind.CLIENT,
        ) as span:
            # Set system identifier
            _safe_set(span, Attr.SYSTEM, "Grok")

            # Set request attributes
            _safe_set(span, Attr.REQ_MODEL, request_model)
            _safe_set(span, Attr.REQ_STREAMING, is_streaming)
            headers = (kwargs or {}).get("extra_headers") or (kwargs or {}).get("headers")
            if headers is not None:
                _safe_set(span, Attr.REQ_HEADERS, str(headers))

            # Extract and set messages
            msg_count, msg_attrs = _extract_messages(messages)
            for key, value in msg_attrs.items():
                _safe_set(span, key, value)
            record_message_count(msg_count)

            # Extract and set tools
            tool_count, tool_attrs = _extract_tools(tools)
            for key, value in tool_attrs.items():
                _safe_set(span, key, value)

            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)

                if is_streaming:
                    # Handle streaming response
                    return await _wrap_stream_response_async(result, span, start_time)
                else:
                    # Handle non-streaming response
                    out = _process_non_stream_response(result, span)
                    span.set_status(Status(StatusCode.OK))
                    return out

            except Exception as e:
                # Optional: record_operation_exception(kind="chat")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return _async_wrapper()


def _wrap_embeddings_create(wrapped, instance, args, kwargs):
    """Wrap OpenAI-style embeddings create method for Grok detection."""
    if not _is_grok_client(instance):
        return wrapped(*args, **kwargs)

    with TRACER.start_as_current_span(
        "embeddings.create",
        kind=SpanKind.CLIENT,
    ) as span:
        # Set system identifier
        _safe_set(span, Attr.SYSTEM, "Grok")

        try:
            result = wrapped(*args, **kwargs)

            # Extract embedding dimensions if available
            if hasattr(result, 'data') and result.data:
                first_embedding = result.data[0]
                if hasattr(first_embedding, 'embedding'):
                    vector_size = len(first_embedding.embedding)
                    record_embedding_vector_size(vector_size)

            span.set_status(Status(StatusCode.OK))
            return result

        except Exception as e:
            record_operation_exception(kind="embeddings")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def instrument_grok():
    """Instrument OpenAI client for Grok (xAI) tracking.

    Auto-detects Grok when base_url contains 'api.x.ai'.
    Can be overridden with LLM_PROVIDER=grok environment variable.
    """
    # Chat completions (sync and async)
    wrap_function_wrapper(
        "openai.resources.chat.completions",
        "Completions.create",
        _wrap_chat_completions_create,
    )

    # Async chat completions
    wrap_function_wrapper(
        "openai.resources.chat.completions",
        "AsyncCompletions.create",
        _wrap_chat_completions_create_async,
    )

    # Responses API (sync/async)
    try:
        wrap_function_wrapper(
            "openai.resources.responses",
            "Responses.create",
            _wrap_chat_completions_create,
        )
        wrap_function_wrapper(
            "openai.resources.responses",
            "AsyncResponses.create",
            _wrap_chat_completions_create_async,
        )
    except Exception:
        pass

    # Embeddings (optional)
    wrap_function_wrapper(
        "openai.resources.embeddings",
        "Embeddings.create",
        _wrap_embeddings_create,
    )

    # Async embeddings
    wrap_function_wrapper(
        "openai.resources.embeddings",
        "AsyncEmbeddings.create",
        _wrap_embeddings_create,
    )


def uninstrument_grok():
    """Remove Grok instrumentation from OpenAI client."""
    # Chat completions
    otel_unwrap(
        "openai.resources.chat.completions",
        "Completions.create",
    )

    # Async chat completions
    otel_unwrap(
        "openai.resources.chat.completions",
        "AsyncCompletions.create",
    )

    # Responses API
    try:
        otel_unwrap(
            "openai.resources.responses",
            "Responses.create",
        )
        otel_unwrap(
            "openai.resources.responses",
            "AsyncResponses.create",
        )
    except Exception:
        pass

    # Embeddings
    otel_unwrap(
        "openai.resources.embeddings",
        "Embeddings.create",
    )

    # Async embeddings
    otel_unwrap(
        "openai.resources.embeddings",
        "AsyncEmbeddings.create",
    )
