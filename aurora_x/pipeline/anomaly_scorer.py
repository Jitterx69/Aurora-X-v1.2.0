"""
AURORA-X Anomaly Pre-Scorer.

Z-score and Isolation Forest-based anomaly pre-scoring for
fast filtering before full fault detection.
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger("aurora_x.pipeline.anomaly_scorer")


class ZScoreScorer:
    """Online Z-score anomaly scorer with adaptive baseline."""

    def __init__(self, window_size: int = 500, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def score(self, feature_name: str, value: float) -> Dict[str, float]:
        """Score a single feature value and return anomaly info."""
        history = self._history[feature_name]
        history.append(value)

        if len(history) < 10:
            return {"z_score": 0.0, "is_anomaly": False, "threshold": self.threshold}

        arr = np.array(history)
        mean = np.mean(arr)
        std = np.std(arr)

        if std < 1e-10:
            z = 0.0
        else:
            z = abs(value - mean) / std

        return {
            "z_score": float(z),
            "is_anomaly": z > self.threshold,
            "threshold": self.threshold,
            "baseline_mean": float(mean),
            "baseline_std": float(std),
        }


class IsolationForestScorer:
    """Lightweight isolation-forest-inspired anomaly scorer.

    Uses a simplified version that computes isolation depth
    via random binary splits on feature history.
    """

    def __init__(self, n_trees: int = 50, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self._history: deque = deque(maxlen=2000)
        self._rng = np.random.RandomState(42)

    def _isolation_depth(self, value: float, data: np.ndarray, max_depth: int = 10) -> float:
        """Compute average isolation depth for a value."""
        depths = []
        for _ in range(self.n_trees):
            sample_idx = self._rng.choice(len(data), size=min(self.sample_size, len(data)), replace=False)
            sample = data[sample_idx]
            depth = self._isolate(value, sample, 0, max_depth)
            depths.append(depth)
        return float(np.mean(depths))

    def _isolate(self, value: float, data: np.ndarray, current_depth: int, max_depth: int) -> int:
        """Recursively isolate a point."""
        if current_depth >= max_depth or len(data) <= 1:
            return current_depth

        min_val, max_val = np.min(data), np.max(data)
        if min_val == max_val:
            return current_depth

        split = self._rng.uniform(min_val, max_val)

        if value < split:
            subset = data[data < split]
        else:
            subset = data[data >= split]

        if len(subset) == 0:
            return current_depth

        return self._isolate(value, subset, current_depth + 1, max_depth)

    def score(self, value: float) -> float:
        """Return anomaly score in [0, 1] where higher = more anomalous."""
        self._history.append(value)

        if len(self._history) < 50:
            return 0.0

        data = np.array(self._history)
        avg_depth = self._isolation_depth(value, data)
        max_depth = 10  # matches max_depth in _isolation_depth

        # Normalize: shorter depth = more anomalous
        score = 1.0 - (avg_depth / max_depth)
        return float(np.clip(score, 0.0, 1.0))


class AnomalyScorer:
    """Unified anomaly pre-scorer that combines multiple methods."""

    def __init__(self, config: Dict[str, Any]):
        method = config.get("anomaly_scorer", {}).get("method", "zscore")
        threshold = config.get("anomaly_scorer", {}).get("threshold", 3.0)

        self.z_scorer = ZScoreScorer(threshold=threshold)
        self._isolation_scorers: Dict[str, IsolationForestScorer] = {}
        self.method = method

        logger.info("AnomalyScorer initialized (method=%s, threshold=%.1f)", method, threshold)

    def score_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Score all features and return anomaly report."""
        anomaly_scores = {}
        max_score = 0.0
        anomalous_features = []

        for name, value in features.items():
            if not isinstance(value, (int, float)):
                continue

            z_result = self.z_scorer.score(name, value)

            if z_result["is_anomaly"]:
                anomalous_features.append(name)

            anomaly_scores[name] = z_result["z_score"]
            max_score = max(max_score, z_result["z_score"])

        return {
            "max_anomaly_score": max_score,
            "anomalous_features": anomalous_features,
            "n_anomalous": len(anomalous_features),
            "is_anomalous": len(anomalous_features) > 0,
            "scores": anomaly_scores,
        }
