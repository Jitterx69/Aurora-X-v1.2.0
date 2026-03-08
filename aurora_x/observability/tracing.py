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
                import os
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource

                resource = Resource.create({
                    "service.name": "aurora-x-python-orchestrator",
                    "service.version": "1.0.0",
                })

                provider = TracerProvider(resource=resource)
                otlp_endpoint = os.getenv("JAERGER_OTLP_ENDPOINT", "http://jaeger:4317")
                
                processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)
                
                self._tracer = trace.get_tracer("aurora_x")
                
                # --- Initialize Rust Core Tracing ---
                try:
                    import aurora_core
                    aurora_core.init_tracing(otlp_endpoint)
                    logger.info("Rust Core tracing initialized")
                except (ImportError, AttributeError) as e:
                    logger.warning("Could not initialize Rust Core tracing: %s", e)

                logger.info("OpenTelemetry tracing enabled (endpoint: %s)", otlp_endpoint)
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
