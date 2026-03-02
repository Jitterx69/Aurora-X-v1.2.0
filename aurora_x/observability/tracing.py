"""
AURORA-X OpenTelemetry Tracing.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("aurora_x.observability.tracing")


class TracingProvider:
    """OpenTelemetry distributed tracing wrapper."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("tracing", {}).get("enabled", False)
        self._tracer = None

        if self.enabled:
            try:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

                provider = TracerProvider()
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer("aurora_x")
                logger.info("OpenTelemetry tracing enabled")
            except ImportError:
                logger.warning("OpenTelemetry not available. Tracing disabled.")
                self.enabled = False
        else:
            logger.info("Tracing disabled")

    def start_span(self, name: str):
        if self._tracer:
            return self._tracer.start_as_current_span(name)
        return _NoOpSpan()


class _NoOpSpan:
    """No-op context manager when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass
