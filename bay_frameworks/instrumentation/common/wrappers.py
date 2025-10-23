from typing import Callable, Any
from dataclasses import dataclass

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import unwrap as otel_unwrap
from opentelemetry.trace import Tracer, Span
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper


_SUPPRESS_INSTRUMENTATION_KEY = "suppress_language_model_instrumentation"


@dataclass
class WrapConfig:
	package: str
	class_name: str
	method_name: str
	trace_name: str
	span_kind: Any
	is_async: bool = False
	# handler is a callable that can accept (args, kwargs, return_value) and returns attributes dict
	handler: Callable[..., dict] = lambda *a, **k: {}


def _finish_span_success(span: Span) -> None:
	span.set_status(Status(StatusCode.OK))


def _finish_span_error(span: Span, exception: Exception) -> None:
	span.record_exception(exception)
	span.set_status(Status(StatusCode.ERROR, str(exception)))


def _update_span(span: Span, attributes: dict) -> None:
	for key, value in (attributes or {}).items():
		try:
			span.set_attribute(key, value)
		except Exception:
			pass


def _with_tracer_wrapper(tracer: Tracer, trace_name: str, span_kind: Any, handler: Callable[..., dict]):
	def wrapper(wrapped, instance, args, kwargs):
		return_value = None
		with tracer.start_as_current_span(trace_name, kind=span_kind) as span:
			try:
				attributes = handler(args=args, kwargs=kwargs)
				_update_span(span, attributes)
				return_value = wrapped(*args, **kwargs)
				attributes = handler(return_value=return_value)
				_update_span(span, attributes)
				_finish_span_success(span)
			except Exception as e:
				attributes = handler(args=args, kwargs=kwargs, return_value=return_value)
				_update_span(span, attributes)
				_finish_span_error(span, e)
				raise
		return return_value

	return wrapper


def unwrap(wrap_config: WrapConfig) -> None:
	otel_unwrap(wrap_config.package, f"{wrap_config.class_name}.{wrap_config.method_name}")


def wrap(wrap_config: WrapConfig, tracer: Tracer) -> Callable:
	handler = wrap_config.handler

	async def awrapper(wrapped, instance, args, kwargs):
		if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
			return await wrapped(*args, **kwargs)
		return_value = None
		with tracer.start_as_current_span(wrap_config.trace_name, kind=wrap_config.span_kind) as span:
			try:
				attributes = handler(args=args, kwargs=kwargs)
				_update_span(span, attributes)
				return_value = await wrapped(*args, **kwargs)
				attributes = handler(return_value=return_value)
				_update_span(span, attributes)
				_finish_span_success(span)
			except Exception as e:
				attributes = handler(args=args, kwargs=kwargs, return_value=return_value)
				_update_span(span, attributes)
				_finish_span_error(span, e)
				raise
		return return_value

	def wrapper(wrapped, instance, args, kwargs):
		if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
			return wrapped(*args, **kwargs)
		return_value = None
		with tracer.start_as_current_span(wrap_config.trace_name, kind=wrap_config.span_kind) as span:
			try:
				attributes = handler(args=args, kwargs=kwargs)
				_update_span(span, attributes)
				return_value = wrapped(*args, **kwargs)
				attributes = handler(return_value=return_value)
				_update_span(span, attributes)
				_finish_span_success(span)
			except Exception as e:
				attributes = handler(args=args, kwargs=kwargs, return_value=return_value)
				_update_span(span, attributes)
				_finish_span_error(span, e)
				raise
		return return_value

	target = awrapper if wrap_config.is_async else wrapper
	wrap_function_wrapper(wrap_config.package, f"{wrap_config.class_name}.{wrap_config.method_name}", target)
	return target


def wrap_function_wrapper(package: str, attribute: str, wrapper: Callable) -> None:  # type: ignore[no-redef]
    """Expose wrapt.wrap_function_wrapper for convenience under common namespace."""
    from wrapt import wrap_function_wrapper as _wfw

    _wfw(package, attribute, wrapper)

