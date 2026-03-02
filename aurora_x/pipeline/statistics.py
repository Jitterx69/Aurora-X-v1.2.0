"""
AURORA-X Rolling Statistics Module.

Computes running statistical features over configurable windows
using numerically-stable single-pass algorithms.
"""

import numpy as np
from typing import Dict, Optional
from collections import deque


class RollingStatistics:
    """Numerically stable rolling statistics using Welford's algorithm."""

    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._m3 = 0.0
        self._m4 = 0.0

    def update(self, value: float):
        """Add a new value to the rolling window."""
        self._buffer.append(value)
        self._count = len(self._buffer)

    def compute(self) -> Dict[str, float]:
        """Compute all statistics on the current window."""
        if self._count == 0:
            return self._empty_stats()

        arr = np.array(self._buffer)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        stats = {
            "mean": mean,
            "std": std,
            "variance": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "rms": float(np.sqrt(np.mean(arr ** 2))),
            "peak": float(np.max(np.abs(arr))),
            "count": n,
        }

        # Crest factor
        rms = stats["rms"]
        stats["crest_factor"] = stats["peak"] / rms if rms > 1e-10 else 0.0

        # Higher-order moments
        if std > 1e-10:
            normalized = (arr - mean) / std
            stats["skewness"] = float(np.mean(normalized ** 3))
            stats["kurtosis"] = float(np.mean(normalized ** 4))
        else:
            stats["skewness"] = 0.0
            stats["kurtosis"] = 0.0

        # Zero-crossing rate
        signs = np.sign(arr - mean)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        stats["zero_crossing_rate"] = float(crossings / max(n - 1, 1))

        # Percentiles
        stats["p25"] = float(np.percentile(arr, 25))
        stats["median"] = float(np.median(arr))
        stats["p75"] = float(np.percentile(arr, 75))
        stats["iqr"] = stats["p75"] - stats["p25"]

        return stats

    def _empty_stats(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "mean", "std", "variance", "min", "max", "range",
            "rms", "peak", "count", "crest_factor", "skewness",
            "kurtosis", "zero_crossing_rate", "p25", "median", "p75", "iqr",
        ]}

    @property
    def is_ready(self) -> bool:
        return self._count >= self.window_size


class MultiChannelStatistics:
    """Manages rolling statistics across multiple sensor channels."""

    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        self._channels: Dict[str, RollingStatistics] = {}

    def update(self, channel: str, value: float):
        """Update a specific channel with a new value."""
        if channel not in self._channels:
            self._channels[channel] = RollingStatistics(self.window_size)
        self._channels[channel].update(value)

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all channels."""
        return {
            channel: stats.compute()
            for channel, stats in self._channels.items()
        }

    def compute_flat(self) -> Dict[str, float]:
        """Compute statistics and return as a flat dict with channel prefixes."""
        result = {}
        for channel, stats in self._channels.items():
            for key, val in stats.compute().items():
                result[f"{channel}_{key}"] = val
        return result
