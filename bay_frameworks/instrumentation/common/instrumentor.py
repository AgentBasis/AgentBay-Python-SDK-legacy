from abc import ABC, abstractmethod
from typing import Collection, Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import Tracer, get_tracer
from opentelemetry.metrics import Meter, get_meter

from agentbay.bay_frameworks.instrumentation.common.wrappers import WrapConfig, wrap, unwrap
from agentbay.bay_frameworks.logging import logger


@dataclass
class InstrumentorConfig:
	"""Configuration for an instrumentor."""

	library_name: str
	library_version: str
	wrapped_methods: List[WrapConfig] = field(default_factory=list)
	metrics_enabled: bool = True
	dependencies: Collection[str] = field(default_factory=list)


class CommonInstrumentor(BaseInstrumentor, ABC):
	"""Base class for bay_frameworks instrumentors with common functionality."""

	def __init__(self, config: InstrumentorConfig):
		super().__init__()
		self.config = config
		self._tracer: Optional[Tracer] = None
		self._meter: Optional[Meter] = None
		self._metrics: Dict[str, Any] = {}

	def instrumentation_dependencies(self) -> Collection[str]:
		return self.config.dependencies

	def _instrument(self, **kwargs):
		tracer_provider = kwargs.get("tracer_provider")
		self._tracer = get_tracer(self.config.library_name, self.config.library_version, tracer_provider)

		if self.config.metrics_enabled:
			meter_provider = kwargs.get("meter_provider")
			self._meter = get_meter(self.config.library_name, self.config.library_version, meter_provider)
			self._metrics = self._create_metrics(self._meter)

		self._initialize(**kwargs)
		self._wrap_methods()
		self._custom_wrap(**kwargs)

	def _uninstrument(self, **kwargs):
		for wrap_config in self.config.wrapped_methods:
			try:
				unwrap(wrap_config)
			except Exception as e:
				logger.debug(
					f"Failed to unwrap {wrap_config.package}.{wrap_config.class_name}.{wrap_config.method_name}: {e}"
				)
		self._custom_unwrap(**kwargs)
		self._tracer = None
		self._meter = None
		self._metrics.clear()

	def _wrap_methods(self):
		for wrap_config in self.config.wrapped_methods:
			try:
				wrap(wrap_config, self._tracer)
			except (AttributeError, ModuleNotFoundError) as e:
				logger.debug(
					f"Could not wrap {wrap_config.package}.{wrap_config.class_name}.{wrap_config.method_name}: {e}"
				)

	@abstractmethod
	def _create_metrics(self, meter: Meter) -> Dict[str, Any]:
		pass

	def _initialize(self, **kwargs):
		pass

	def _custom_wrap(self, **kwargs):
		pass

	def _custom_unwrap(self, **kwargs):
		pass


def create_wrapper_factory(wrapper_func: Callable, *wrapper_args, **wrapper_kwargs) -> Callable:
	def factory(tracer: Tracer):
		def wrapper(wrapped, instance, args, kwargs):
			return wrapper_func(
				tracer, *wrapper_args, wrapped=wrapped, instance=instance, args=args, kwargs=kwargs, **wrapper_kwargs
			)

		return wrapper

	return factory


