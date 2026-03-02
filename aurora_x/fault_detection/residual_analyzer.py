"""
AURORA-X Residual Analyzer.

Physics-model residual computation with adaptive threshold detection
for the first layer of fault detection.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

logger = logging.getLogger("aurora_x.fault_detection.residual")


class AdaptiveThreshold:
    """Adaptive threshold that adjusts based on signal statistics."""

    def __init__(self, window_size: int = 500, multiplier: float = 2.5):
        self.window_size = window_size
        self.multiplier = multiplier
        self._history: deque = deque(maxlen=window_size)

    def update_and_check(self, value: float) -> Dict[str, Any]:
        self._history.append(abs(value))

        if len(self._history) < 20:
            return {"exceeded": False, "threshold": 0.0, "value": value}

        arr = np.array(self._history)
        mean = np.mean(arr)
        std = np.std(arr)
        threshold = mean + self.multiplier * std

        return {
            "exceeded": abs(value) > threshold,
            "threshold": float(threshold),
            "value": float(value),
            "mean": float(mean),
            "std": float(std),
            "normalized": float((abs(value) - mean) / (std + 1e-10)),
        }


class ResidualAnalyzer:
    """Computes residuals between expected (physics/KF) and observed states."""

    def __init__(self, config: Dict[str, Any]):
        self.threshold_multiplier = config.get("residual_threshold", 2.5)
        self._thresholds: Dict[str, AdaptiveThreshold] = defaultdict(
            lambda: AdaptiveThreshold(multiplier=self.threshold_multiplier)
        )
        logger.info("ResidualAnalyzer initialized (threshold=%.1fx)", self.threshold_multiplier)

    def analyze(
        self,
        asset_id: str,
        state: Dict[str, Any],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute residuals and check thresholds.

        Compares the KF-estimated state against raw feature values
        to detect discrepancies indicating faults.
        """
        residuals = {}
        alerts = []

        # Key signals to check
        state_keys = ["temperature", "vibration", "pressure", "flow", "electrical", "acoustic"]
        feat_dict = features.get("features", features)

        for key in state_keys:
            state_val = state.get(key)
            # Find matching feature
            feature_val = None
            for pattern in [f"{key}_rolling_mean", f"{key}_rms", f"{key}_mean"]:
                if pattern in feat_dict:
                    feature_val = feat_dict[pattern]
                    break

            if state_val is None or feature_val is None:
                continue

            # Compute residual
            residual = float(feature_val - state_val)
            thresh_key = f"{asset_id}:{key}"
            check = self._thresholds[thresh_key].update_and_check(residual)

            residuals[key] = {
                "residual": residual,
                "abs_residual": abs(residual),
                **check,
            }

            if check["exceeded"]:
                alerts.append({
                    "signal": key,
                    "residual": residual,
                    "threshold": check["threshold"],
                    "severity": min(1.0, check["normalized"] / 5.0),
                })

        return {
            "asset_id": asset_id,
            "residuals": residuals,
            "alerts": alerts,
            "n_alerts": len(alerts),
            "max_severity": max((a["severity"] for a in alerts), default=0.0),
        }
