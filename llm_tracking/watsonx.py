"""IBM watsonx.ai instrumentation 

Wraps common ModelInference methods to capture model, prompts/completions, usage, and streaming timings.
"""

from __future__ import annotations

import json
import time
import inspect
from typing import Any, Dict, Optional

from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry.instrumentation.utils import unwrap as otel_unwrap

from .config import maybe_redact
from .metrics import record_message_count
from .metrics import record_embedding_vector_size
from .metrics import record_streaming_metrics, record_operation_exception, record_message_event


TRACER = get_tracer("llmtracker.watsonx")


class Attr:
    SYSTEM = "gen_ai.system"
    REQ_MODEL = "gen_ai.request.model"
    REQ_STREAMING = "gen_ai.request.streaming"
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


def _set_prompt_from_kwargs(span, kwargs: Dict[str, Any]):
    # watsonx generate/chat accept input via 'input' or 'messages'
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        try:
            record_message_count(len(kwargs["messages"]))
            _safe_set(span, "gen_ai.prompt.count", len(kwargs["messages"]))
        except Exception:
            pass
        for i, msg in enumerate(kwargs["messages"]):
            d = msg if isinstance(msg, dict) else _model_as_dict(msg)
            role = d.get("role", "user")
            content = d.get("content") or d.get("text")
            try:
                record_message_event(role=role or "user", provider="watsonx")
            except Exception:
                pass
            red = maybe_redact(content) if content is not None else None
            _safe_set(span, f"{Attr.PROMPT}.{i}.role", role)
            if red is not None:
                _safe_set(span, f"{Attr.PROMPT}.{i}.content", red)
    elif "input" in kwargs:
        try:
            record_message_count(1)
            _safe_set(span, "gen_ai.prompt.count", 1)
        except Exception:
            pass
        try:
            record_message_event(role="user", provider="watsonx")
        except Exception:
            pass
        red = maybe_redact(str(kwargs.get("input")))
        _safe_set(span, f"{Attr.PROMPT}.0.role", "user")
        if red is not None:
            _safe_set(span, f"{Attr.PROMPT}.0.content", red)


def _set_usage(span, response: Any):
    d = _model_as_dict(response)
    usage = d.get("usage") or d.get("token_usage")
    if isinstance(usage, dict):
        # Attempt common fields
        pt = usage.get("input_tokens") or usage.get("prompt_tokens")
        ct = usage.get("output_tokens") or usage.get("completion_tokens")
        tt = usage.get("total_tokens")
        if pt is not None:
            _safe_set(span, Attr.USE_PROMPT, pt)
        if ct is not None:
            _safe_set(span, Attr.USE_COMP, ct)
        if tt is None and pt is not None and ct is not None:
            tt = pt + ct
        if tt is not None:
            _safe_set(span, Attr.USE_TOTAL, tt)


def _set_completion(span, response: Any):
    d = _model_as_dict(response)
    if "model_id" in d:
        _safe_set(span, Attr.RESP_MODEL, d.get("model_id"))
    if "id" in d:
        _safe_set(span, Attr.RESP_ID, d.get("id"))
    # Content
    text = d.get("generated_text") or d.get("output_text") or d.get("text")
    if text:
        red = maybe_redact(str(text))
        _safe_set(span, f"{Attr.COMPLETION}.0.role", "assistant")
        if red is not None:
            _safe_set(span, f"{Attr.COMPLETION}.0.content", red)
    _set_usage(span, response)


def _set_embedding_info(span, response: Any):
    d = _model_as_dict(response)
    size = None
    try:
        if isinstance(d.get("embedding"), list):
            size = len(d.get("embedding"))
        elif isinstance(d.get("vector"), list):
            size = len(d.get("vector"))
        elif isinstance(d.get("data"), list) and d.get("data"):
            first = d["data"][0]
            fd = first if isinstance(first, dict) else _model_as_dict(first)
            if isinstance(fd.get("embedding"), list):
                size = len(fd.get("embedding"))
    except Exception:
        size = None
    if isinstance(size, int) and size > 0:
        _safe_set(span, "gen_ai.embeddings.vector_size", size)
        try:
            record_embedding_vector_size(size)
        except Exception:
            pass


def _wrap_method(trace_name: str, is_async: bool = False):
    def sync_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "watsonx"}
        model = None
        try:
            model = getattr(instance, "model_id", None)
        except Exception:
            pass
        if isinstance(kwargs, dict) and not model:
            model = kwargs.get("model_id") or kwargs.get("model")
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _set_prompt_from_kwargs(span, kwargs or {})
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    if isinstance(trace_name, str) and "embed" in trace_name.lower():
                        record_operation_exception("embeddings")
                except Exception:
                    pass
                raise
            try:
                _set_completion(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    async def async_wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "watsonx"}
        model = None
        try:
            model = getattr(instance, "model_id", None)
        except Exception:
            pass
        if isinstance(kwargs, dict) and not model:
            model = kwargs.get("model_id") or kwargs.get("model")
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _set_prompt_from_kwargs(span, kwargs or {})
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    if isinstance(trace_name, str) and "embed" in trace_name.lower():
                        record_operation_exception("embeddings")
                except Exception:
                    pass
                raise
            try:
                _set_completion(span, result)
                span.set_status(Status(StatusCode.OK))
            finally:
                pass
            return result

    return async_wrapper if is_async else sync_wrapper


def _wrap_stream_method(trace_name: str):
    def wrapper(wrapped, instance, args, kwargs):
        attributes = {Attr.SYSTEM: "watsonx"}
        model = getattr(instance, "model_id", None)
        with TRACER.start_as_current_span(trace_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
            _safe_set(span, Attr.REQ_MODEL, model)
            _safe_set(span, Attr.REQ_STREAMING, True)
            _set_prompt_from_kwargs(span, kwargs or {})
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
                    span.set_status(Status(StatusCode.OK))
                    span.end()

            return generator()

    return wrapper


def instrument_watsonx() -> None:
    """Enable automatic tracking for IBM watsonx.ai API calls."""
    try:
        base = "ibm_watsonx_ai.foundation_models.inference"
        wrap_function_wrapper(base, "ModelInference.generate", _wrap_method("watsonx.generate"))
        wrap_function_wrapper(base, "ModelInference.chat", _wrap_method("watsonx.chat"))
        wrap_function_wrapper(base, "ModelInference.tokenize", _wrap_method("watsonx.tokenize"))
        wrap_function_wrapper(base, "ModelInference.get_details", _wrap_method("watsonx.get_details"))
        # Streams
        wrap_function_wrapper(base, "ModelInference.generate_text_stream", _wrap_stream_method("watsonx.generate_text_stream"))
        wrap_function_wrapper(base, "ModelInference.chat_stream", _wrap_stream_method("watsonx.chat_stream"))
        # Optional embeddings surfaces (best-effort)
        try:
            def _wrap_embed(trace):
                def handler(wrapped, instance, args, kwargs):
                    with TRACER.start_as_current_span(trace, kind=SpanKind.CLIENT, attributes={Attr.SYSTEM: "watsonx"}) as span:
                        try:
                            result = wrapped(*args, **kwargs)
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise
                        _set_embedding_info(span, result)
                        span.set_status(Status(StatusCode.OK))
                        return result
                return handler

            wrap_function_wrapper(base, "ModelInference.embed", _wrap_embed("watsonx.embed"))
        except Exception:
            pass
    except Exception:
        pass


def uninstrument_watsonx() -> None:
    """Disable IBM watsonx.ai tracking."""
    targets = [
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.generate"),
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.chat"),
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.tokenize"),
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.get_details"),
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.generate_text_stream"),
        ("ibm_watsonx_ai.foundation_models.inference", "ModelInference.chat_stream"),
    ]
    for mod, meth in targets:
        try:
            otel_unwrap(mod, meth)
        except Exception:
            pass
