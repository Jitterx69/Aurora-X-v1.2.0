"""
AURORA-X Edge Gateway.

Performs schema validation, timestamp correction, compression,
and preliminary feature extraction on incoming sensor events.
"""

import time
import math
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("aurora_x.ingestion.gateway")

# Required fields for a valid sensor event
REQUIRED_FIELDS = {"asset_id", "timestamp", "sensors"}


class EdgeGateway:
    """Edge gateway for sensor data validation and preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.enable_compression = config.get("edge_gateway", {}).get(
            "enable_compression", True
        )
        self.timestamp_tolerance_ms = config.get("edge_gateway", {}).get(
            "timestamp_tolerance_ms", 500
        )

        # Tracking
        self.events_processed = 0
        self.events_rejected = 0
        self._feature_buffers: Dict[str, List[float]] = {}

        logger.info("Edge gateway initialized (compression=%s)", self.enable_compression)

    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate, correct, and enrich a raw sensor event.

        Returns enriched event or None if validation fails.
        Publishing to broker/log is handled by the orchestrator.
        """
        # 1. Schema validation
        if not self._validate_schema(event):
            self.events_rejected += 1
            return None

        # 2. Timestamp correction
        event = self._correct_timestamp(event)

        # 3. Preliminary feature extraction
        enriched = self._extract_features(event)

        # 4. Generate event fingerprint
        enriched["fingerprint"] = self._fingerprint(enriched)

        self.events_processed += 1
        return enriched

    def _validate_schema(self, event: Dict[str, Any]) -> bool:
        """Validate required fields and data types."""
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in event:
                logger.warning("Event missing required field: %s", field)
                return False

        # Validate sensors is a dict
        if not isinstance(event.get("sensors"), dict):
            logger.warning("Event 'sensors' field is not a dict")
            return False

        # Validate sensor values are numeric
        for sensor, value in event["sensors"].items():
            if not isinstance(value, (int, float)):
                logger.warning("Sensor '%s' has non-numeric value: %s", sensor, value)
                return False
            if math.isnan(value) or math.isinf(value):
                logger.warning("Sensor '%s' has nan/inf value", sensor)
                return False

        return True

    def _correct_timestamp(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply NTP-style timestamp correction."""
        now = time.time()
        event_time = event.get("timestamp", now)

        drift_ms = abs(now - event_time) * 1000

        if drift_ms > self.timestamp_tolerance_ms:
            logger.debug(
                "Timestamp drift %.1fms on %s, correcting",
                drift_ms, event.get("asset_id"),
            )
            event["original_timestamp"] = event_time
            event["timestamp"] = now
            event["timestamp_corrected"] = True
        else:
            event["timestamp_corrected"] = False

        return event

    def _extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract preliminary features (RMS, peak, kurtosis) from sensor data."""
        asset_id = event["asset_id"]
        features = {}

        for sensor_name, value in event.get("sensors", {}).items():
            buf_key = f"{asset_id}:{sensor_name}"

            if buf_key not in self._feature_buffers:
                self._feature_buffers[buf_key] = []

            buf = self._feature_buffers[buf_key]
            buf.append(value)

            # Keep last 64 samples for feature computation
            if len(buf) > 64:
                buf.pop(0)

            if len(buf) >= 8:
                arr = np.array(buf)
                features[f"{sensor_name}_rms"] = float(np.sqrt(np.mean(arr ** 2)))
                features[f"{sensor_name}_peak"] = float(np.max(np.abs(arr)))
                features[f"{sensor_name}_mean"] = float(np.mean(arr))
                if np.std(arr) > 0:
                    features[f"{sensor_name}_kurtosis"] = float(
                        np.mean(((arr - np.mean(arr)) / np.std(arr)) ** 4)
                    )
                else:
                    features[f"{sensor_name}_kurtosis"] = 0.0

        event["edge_features"] = features
        return event

    def _fingerprint(self, event: Dict[str, Any]) -> str:
        """Generate a unique fingerprint for deduplication."""
        content = f"{event['asset_id']}:{event['timestamp']}:{json.dumps(event.get('sensors', {}), sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, int]:
        return {
            "events_processed": self.events_processed,
            "events_rejected": self.events_rejected,
        }
