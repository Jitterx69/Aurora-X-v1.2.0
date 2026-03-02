"""
AURORA-X Prometheus Metrics Collector.
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger("aurora_x.observability.metrics")

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False


class MetricsCollector:
    """Collects and exposes Prometheus metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._events_processed = 0
        self._start_time = time.time()

        if PROM_AVAILABLE and config.get("prometheus", {}).get("enabled", False):
            self.events_total = Counter("aurora_events_processed_total", "Total events processed")
            self.processing_latency = Histogram("aurora_processing_latency_seconds", "Processing latency")
            self.active_assets = Gauge("aurora_active_assets", "Number of active assets")
            self.fault_severity = Gauge("aurora_fault_severity", "Current max fault severity", ["asset_id"])
            self.rl_reward = Gauge("aurora_rl_reward", "Latest RL reward")
            self.safety_violations = Counter("aurora_safety_violations_total", "Total safety violations")

            port = config.get("prometheus", {}).get("port", 9090)
            try:
                start_http_server(port)
                logger.info("Prometheus metrics server started on port %d", port)
            except Exception as e:
                logger.warning("Could not start Prometheus server: %s", e)
        else:
            self.events_total = None
            logger.info("Prometheus metrics disabled")

    def record_event_processed(self, latency: float = 0.0):
        self._events_processed += 1
        if self.events_total:
            self.events_total.inc()
        if self.events_total and latency > 0:
            self.processing_latency.observe(latency)

    def record_fault_severity(self, asset_id: str, severity: float):
        if self.events_total:
            self.fault_severity.labels(asset_id=asset_id).set(severity)

    def record_rl_reward(self, reward: float):
        if self.events_total:
            self.rl_reward.set(reward)

    def record_safety_violation(self):
        if self.events_total:
            self.safety_violations.inc()

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "events_processed": self._events_processed,
            "uptime_seconds": uptime,
            "events_per_second": self._events_processed / max(uptime, 1),
        }
