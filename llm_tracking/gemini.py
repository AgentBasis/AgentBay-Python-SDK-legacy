"""Google GenAI (Gemini) instrumentation 

Covers both `google.genai` (new SDK) and `google.generativeai` (classic SDK) where possible.

Tracks (read-only):
- Model + response IDs (when available)
- Conversation: prompts from inputs; completions from responses
- Usage: prompt/completion/total tokens (from usage_metadata or counts)
- Streaming (google.genai stream APIs): first token time, generation time, chunk count

Does NOT record control parameters.
"""

from __future__ import annotations

import json
import time
import inspect
from typing import Any, Dict, Iterable, List, Optional

from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry.instrumentation.utils import unwrap as otel_unwrap

from .config import maybe_redact
from .metrics import (
    record_token_usage,
    record_streaming_metrics,
    record_choice_count,
    record_embedding_vector_size,
    record_operation_exception,
    record_message_count,
    record_tool_call_count,
    record_tool_call_event,
    record_message_event,
)


TRACER = get_tracer("llmtracker.gemini")


class Attr:
    SYSTEM = "gen_ai.system"
    REQ_MODEL = "gen_ai.request.model"
    REQ_HEADERS = "gen_ai.request.headers"
    REQ_STREAMING = "gen_ai.request.streaming"
    REQ_FUNCTIONS = "gen_ai.request.tools"
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"
    RESP_ID = "gen_ai.response.id"
    RESP_MODEL = "gen_ai.response.model"
    USE_PROMPT = "gen_ai.usage.prompt_tokens"
    USE_COMP = "gen_ai.usage.completion_tokens"
    USE_TOTAL = "gen_ai.usage.total_tokens"
    STREAM_TTFB = "gen_ai.streaming.time_to_first_token"
    STREAM_TTG = "gen_ai.streaming.time_to_generate"
    STREAM_CHUNKS = "gen_ai.streaming.chunk_count"
    CHOICE_COUNT = "gen_ai.completion.choice_count"
    EMB_VECTOR_SIZE = "gen_ai.embeddings.vector_size"


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
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {}


def _extract_text_from_input(input_data: Any) -> str:
    # Accept string or list of parts
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        parts = []
        for p in input_data:
            if isinstance(p, str):
                parts.append(p)
            else:
                d = _model_as_dict(p)
                if "text" in d:
                    parts.append(str(d["text"]))
        return " ".join(parts)
    return ""


def _maybe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_grpc_contents(request_like: Any) -> Iterable[Any]:
    if request_like is None:
        return []
    contents = _maybe_get(request_like, "contents")
    if contents:
        return contents
    prompt = _maybe_get(request_like, "prompt")
    if prompt:
        return [prompt]
    return []


def _extract_text_from_grpc_part(part: Any) -> str:
    if part is None:
        return ""
    for attr in ("text", "input_text"):
        value = _maybe_get(part, attr)
        if isinstance(value, str):
            return value
    try:
        as_dict = _model_as_dict(part)
        text = as_dict.get("text") or as_dict.get("input_text")
        if isinstance(text, str):
            return text
    except Exception:
        pass
    return ""


def _set_prompts_from_grpc_request(span, request_like: Any):
    contents = list(_iter_grpc_contents(request_like))
    if not contents:
        return

    filtered: List[str] = []
    for content in contents:
        role = _maybe_get(content, "role")
        if isinstance(role, str):
            role_norm = role.lower()
        elif hasattr(role, "name"):
            try:
                role_norm = str(role.name).lower()
            except Exception:
                role_norm = "user"
        elif isinstance(role, int):
            role_norm = {0: "unspecified", 1: "user", 2: "model"}.get(role, "user")
        else:
            role_norm = "user"

        if role_norm not in {"user", "unspecified", ""}:
            continue

        parts = _maybe_get(content, "parts")
        text_segments: List[str] = []
        if isinstance(parts, Iterable) and not isinstance(parts, (str, bytes)):
            for part in parts:
                text = _extract_text_from_grpc_part(part)
                if text:
                    text_segments.append(text)
        else:
            text = _extract_text_from_input(content)
            if text:
                text_segments.append(text)

        merged = " ".join(text_segments).strip()
        if merged:
            filtered.append(merged)

    if not filtered:
        return

    try:
        _safe_set(span, "gen_ai.prompt.count", len(filtered))
        record_message_count(len(filtered))
    except Exception:
        pass

    for idx, text in enumerate(filtered):
        _safe_set(span, f"{Attr.PROMPT}.{idx}.role", "user")
        red = maybe_redact(text)
        if red is not None:
            _safe_set(span, f"{Attr.PROMPT}.{idx}.content", red)
            try:
                record_message_event(role="user", provider="Gemini")
            except Exception:
                pass


def _set_prompts_from_args_kwargs(span, args, kwargs):
    # google.genai.generate_content inputs via kwargs["contents"] or args[1:]
    contents = None
    if isinstance(kwargs, dict):
        contents = kwargs.get("contents") or kwargs.get("input") or kwargs.get("prompt")
    if contents is None and args:
        # Heuristic: first non-self arg could be contents
        possible = [a for a in args if not hasattr(a, "__class__") or a.__class__.__name__ != "Models"]
        contents = possible[0] if possible else None
    if contents is None:
        return
    # Normalize to list
    entries = contents if isinstance(contents, list) else [contents]
    try:
        _safe_set(span, "gen_ai.prompt.count", len(entries))
        record_message_count(len(entries))
    except Exception:
        pass
    for i, entry in enumerate(entries):
        text = _extract_text_from_input(entry)
        try:
            record_message_event(role="user", provider="Gemini")
        except Exception:
            pass
        if text:
            _safe_set(span, f"{Attr.PROMPT}.{i}.role", "user")
            red = maybe_redact(text)
            if red is not None:
                _safe_set(span, f"{Attr.PROMPT}.{i}.content", red)


def _set_usage_from_response(span, response: Any):
    # google.generativeai responses often have usage_metadata
    d = _model_as_dict(response)
    usage = d.get("usage_metadata") or d.get("usage")
    if isinstance(usage, dict):
        pt = usage.get("prompt_token_count")
        ct = usage.get("candidates_token_count") or usage.get("completion_token_count")
        tt = usage.get("total_token_count")
        if pt is not None:
            _safe_set(span, Attr.USE_PROMPT, pt)
        if ct is not None:
            _safe_set(span, Attr.USE_COMP, ct)
        if tt is not None:
            _safe_set(span, Attr.USE_TOTAL, tt)
        try:
            record_token_usage(pt if isinstance(pt, int) else None, ct if isinstance(ct, int) else None)
        except Exception:
            pass


def _set_completion_from_response(span, response: Any):
    d = _model_as_dict(response)
    # Response id/model if present
    if "response_id" in d:
        _safe_set(span, Attr.RESP_ID, d.get("response_id"))
    if "model" in d:
        _safe_set(span, Attr.RESP_MODEL, d.get("model"))
    # Content
    text = None
    if "text" in d and isinstance(d.get("text"), str):
        text = d.get("text")
    elif "candidates" in d and isinstance(d.get("candidates"), list):
        # Record choice count
        try:
            record_choice_count(len(d["candidates"]))
            _safe_set(span, Attr.CHOICE_COUNT, len(d["candidates"]))
        except Exception:
            pass
        tool_calls = 0
        for c in d["candidates"]:
            cd = _model_as_dict(c)
            # candidate -> content -> parts[{text:...}] or aggregate text
            if "content" in cd:
                content = cd["content"]
                if isinstance(content, dict) and "parts" in content:
                    parts = content["parts"]
                    text_parts = []
                    for p in parts:
                        pd = _model_as_dict(p)
                        # Detect tool/function calls in parts
                        func_call = None
                        if isinstance(pd.get("functionCall"), dict):
                            func_call = pd.get("functionCall")
                        elif isinstance(pd.get("function_call"), dict):
                            func_call = pd.get("function_call")
                        if isinstance(func_call, dict):
                            tool_calls += 1
                            try:
                                record_tool_call_event(tool_name=func_call.get("name"), provider="Gemini")
                            except Exception:
                                pass
                            # Create a child span for the tool call
                            try:
                                name = func_call.get("name") if isinstance(func_call.get("name"), str) else "function"
                                args = func_call.get("args") or func_call.get("arguments")
                                arg_str = None
                                if isinstance(args, (dict, list)):
                                    try:
                                        arg_str = json.dumps(args)
                                    except Exception:
                                        arg_str = str(args)
                                elif isinstance(args, str):
                                    arg_str = args
                                with TRACER.start_as_current_span(
                                    name=f"tool_call.{name}", kind=SpanKind.INTERNAL
                                ) as tool_span:
                                    _safe_set(tool_span, Attr.SYSTEM, "Gemini")
                                    _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.name", name)
                                    if arg_str is not None:
                                        _safe_set(tool_span, f"{Attr.REQ_FUNCTIONS}.0.arguments", arg_str)
                            except Exception:
                                pass
                        if "text" in pd:
                            text_parts.append(str(pd.get("text")))
                    if text_parts:
                        text = " ".join(text_parts)
                        break
        # Record tool call count if any
        try:
            if tool_calls > 0:
                _safe_set(span, "gen_ai.completion.tool_call_count", tool_calls)
                record_tool_call_count(tool_calls)
        except Exception:
            pass
    if text:
        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
        red = maybe_redact(text)
        if red is not None:
            _safe_set(span, f"{Attr.COMPLETION}.0.content", red)

    # Token counts (also covers count/compute tokens responses)
    if "total_tokens" in d and d.get("total_tokens") is not None:
        _safe_set(span, Attr.USE_TOTAL, d.get("total_tokens"))
    if "total_token_count" in d and d.get("total_token_count") is not None:
        _safe_set(span, Attr.USE_TOTAL, d.get("total_token_count"))

    _set_usage_from_response(span, response)

    # Embeddings shape (classic embed_content)
    if "embedding" in d:
        emb = d.get("embedding")
        if isinstance(emb, dict) and "values" in emb and isinstance(emb["values"], list):
            size = len(emb["values"])
            _safe_set(span, Attr.EMB_VECTOR_SIZE, size)
            try:
                record_embedding_vector_size(size)
            except Exception:
                pass
    elif "values" in d and isinstance(d.get("values"), list):
        size = len(d.get("values"))
        _safe_set(span, Attr.EMB_VECTOR_SIZE, size)
        try:
            record_embedding_vector_size(size)
        except Exception:
            pass


def _wrap_method(trace_name: str, system_name: str, is_async: bool = False):
    def sync_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: system_name}
        # Model can be on instance.model or kwargs
        model = None
        try:
            model = getattr(instance, "model", None)
        except Exception:
            pass
        if isinstance(kwargs, dict) and not model:
            model = kwargs.get("model")
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _set_prompts_from_args_kwargs(span, args, kwargs)
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    if "embed" in trace_name.lower():
                        record_operation_exception("embeddings")
                except Exception:
                    pass
                raise
            try:
                _set_completion_from_response(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    async def async_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: system_name}
        model = None
        try:
            model = getattr(instance, "model", None)
        except Exception:
            pass
        if isinstance(kwargs, dict) and not model:
            model = kwargs.get("model")
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _set_prompts_from_args_kwargs(span, args, kwargs)
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    if "embed" in trace_name.lower():
                        record_operation_exception("embeddings")
                except Exception:
                    pass
                raise
            try:
                _set_completion_from_response(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    return async_wrapper if is_async else sync_wrapper


def _wrap_stream_method(trace_name: str, system_name: str):
    def wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: system_name}
        model = getattr(instance, "model", None)
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, True)
            _set_prompts_from_args_kwargs(span, args, kwargs)
            start_time = time.time()
            first_token_time: Optional[float] = None
            chunk_count = 0
            try:
                stream_iter = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            def generator():
                nonlocal first_token_time, chunk_count
                try:
                    for chunk in stream_iter:
                        chunk_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                            _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                        yield chunk
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

    return wrapper


def _wrap_grpc_method(trace_name: str, *, streaming: bool = False):
    def wrapper(wrapped, instance, args, kwargs):
        request = None
        if isinstance(kwargs, dict):
            request = kwargs.get("request")
        if request is None and args:
            request = args[0]

        model = None
        if isinstance(kwargs, dict):
            model = kwargs.get("model")
        if model is None:
            model = _maybe_get(request, "model") or getattr(instance, "model", None)

        attributes = {Attr.SYSTEM: "Gemini"}
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            if streaming:
                _safe_set(span, Attr.REQ_STREAMING, True)
            _set_prompts_from_grpc_request(span, request or kwargs)

            start_time = time.time()
            first_token_time: Optional[float] = None
            chunk_count = 0
            last_response: Optional[Any] = None

            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

            if not streaming:
                try:
                    _set_completion_from_response(span, result)
                    span.set_status(Status(StatusCode.OK))
                finally:
                    pass
                return result

            def generator():
                nonlocal first_token_time, chunk_count, last_response
                try:
                    for chunk in result:
                        last_response = chunk
                        chunk_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                            _safe_set(span, Attr.STREAM_TTFB, first_token_time)
                        yield chunk
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
                finally:
                    end_time = time.time()
                    if first_token_time is not None:
                        _safe_set(span, Attr.STREAM_TTG, max(end_time - start_time - first_token_time, 0.0))
                    _safe_set(span, Attr.STREAM_CHUNKS, chunk_count)
                    try:
                        record_streaming_metrics(
                            ttfb=first_token_time,
                            generate_time=(max(end_time - start_time - first_token_time, 0.0) if first_token_time is not None else None),
                            chunks=chunk_count,
                        )
                    except Exception:
                        pass
                    if last_response is not None:
                        try:
                            _set_completion_from_response(span, last_response)
                        except Exception:
                            pass
                    span.set_status(Status(StatusCode.OK))
                    span.end()

            return generator()

    return wrapper


def instrument_gemini() -> None:
    """Enable automatic tracking for Gemini API calls."""
    # New google.genai models surface
    try:
        wrap_function_wrapper("google.genai.models", "Models.generate_content", _wrap_method("gemini.generate_content", "Gemini"))
        wrap_function_wrapper(
            "google.genai.models", "AsyncModels.generate_content", _wrap_method("gemini.generate_content", "Gemini", is_async=True)
        )
        # Token counting
        wrap_function_wrapper("google.genai.models", "Models.count_tokens", _wrap_method("gemini.count_tokens", "Gemini"))
        wrap_function_wrapper(
            "google.genai.models", "AsyncModels.count_tokens", _wrap_method("gemini.count_tokens", "Gemini", is_async=True)
        )
        wrap_function_wrapper("google.genai.models", "Models.compute_tokens", _wrap_method("gemini.compute_tokens", "Gemini"))
        wrap_function_wrapper(
            "google.genai.models", "AsyncModels.compute_tokens", _wrap_method("gemini.compute_tokens", "Gemini", is_async=True)
        )
    except Exception:
        pass
    # Streaming for google.genai
    try:
        wrap_function_wrapper(
            "google.genai.models", "Models.generate_content_stream", _wrap_stream_method("gemini.generate_content_stream", "Gemini")
        )
    except Exception:
        pass

    # Classic google.generativeai GenerativeModel
    try:
        wrap_function_wrapper(
            "google.generativeai", "GenerativeModel.generate_content", _wrap_method("gemini.generate_content", "Gemini")
        )
    except Exception:
        pass

    # Chat sessions (best effort)
    try:
        wrap_function_wrapper(
            "google.generativeai", "ChatSession.send_message", _wrap_method("gemini.chat.send_message", "Gemini")
        )
        # Streaming chat (if available)
        try:
            wrap_function_wrapper(
                "google.generativeai", "ChatSession.send_message_stream", _wrap_stream_method("gemini.chat.send_message_stream", "Gemini")
            )
        except Exception:
            pass
    except Exception:
        pass

    # Classic embeddings (function)
    try:
        wrap_function_wrapper("google.generativeai", "embed_content", _wrap_method("gemini.embed_content", "Gemini"))
    except Exception:
        pass

    # gRPC client (google.ai.generativelanguage_v1beta)
    try:
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceClient.generate_content",
            _wrap_grpc_method("gemini.grpc.generate_content"),
        )
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceClient.stream_generate_content",
            _wrap_grpc_method("gemini.grpc.stream_generate_content", streaming=True),
        )
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceClient.count_tokens",
            _wrap_grpc_method("gemini.grpc.count_tokens"),
        )
    except Exception:
        pass
    try:
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceAsyncClient.generate_content",
            _wrap_grpc_method("gemini.grpc.generate_content"),
        )
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceAsyncClient.stream_generate_content",
            _wrap_grpc_method("gemini.grpc.stream_generate_content", streaming=True),
        )
        wrap_function_wrapper(
            "google.ai.generativelanguage_v1beta.services.generative_service.client",
            "GenerativeServiceAsyncClient.count_tokens",
            _wrap_grpc_method("gemini.grpc.count_tokens"),
        )
    except Exception:
        pass


def uninstrument_gemini() -> None:
    """Disable Gemini tracking."""
    targets = [
        ("google.genai.models", "Models.generate_content"),
        ("google.genai.models", "AsyncModels.generate_content"),
        ("google.genai.models", "Models.count_tokens"),
        ("google.genai.models", "AsyncModels.count_tokens"),
        ("google.genai.models", "Models.compute_tokens"),
        ("google.genai.models", "AsyncModels.compute_tokens"),
        ("google.genai.models", "Models.generate_content_stream"),
        ("google.generativeai", "GenerativeModel.generate_content"),
        ("google.generativeai", "ChatSession.send_message"),
        ("google.generativeai", "ChatSession.send_message_stream"),
        ("google.generativeai", "embed_content"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceClient.generate_content"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceClient.stream_generate_content"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceClient.count_tokens"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceAsyncClient.generate_content"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceAsyncClient.stream_generate_content"),
        ("google.ai.generativelanguage_v1beta.services.generative_service.client", "GenerativeServiceAsyncClient.count_tokens"),
    ]
    for mod, meth in targets:
        try:
            otel_unwrap(mod, meth)
        except Exception:
            pass
