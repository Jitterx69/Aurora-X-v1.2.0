"""
AURORA-X Stream Processor.

Windowed aggregation engine that processes raw sensor events into
structured feature vectors for downstream state estimation and fault detection.

When aurora_core is available, uses Rust ring-buffer implementation.
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger("aurora_x.pipeline.stream_processor")

# ── Try Rust acceleration ──
try:
    from aurora_core import StreamProcessor as RustStreamProcessor
    _HAS_RUST = True
    logger.info("Rust-accelerated StreamProcessor available (aurora_core)")
except ImportError:
    _HAS_RUST = False


class WindowBuffer:
    """Sliding/tumbling window buffer for a single sensor channel."""

    def __init__(self, window_size: int = 256, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self._buffer: deque = deque(maxlen=window_size * 2)
        self._samples_since_emit = 0

    def add(self, value: float) -> Optional[np.ndarray]:
        """Add a sample. Returns a window array when ready."""
        self._buffer.append(value)
        self._samples_since_emit += 1

        if (len(self._buffer) >= self.window_size
                and self._samples_since_emit >= self.step_size):
            self._samples_since_emit = 0
            return np.array(list(self._buffer)[-self.window_size:])
        return None


class StreamProcessor:
    """Processes raw sensor events through windowed feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        self.window_size = config.get("window_size_samples", 256)
        self.overlap = config.get("window_overlap", 0.5)
        self.fft_bins = config.get("fft_n_bins", 128)
        self.rolling_window = config.get("rolling_stats_window", 64)

        # Per-asset, per-sensor window buffers
        self._buffers: Dict[str, Dict[str, WindowBuffer]] = defaultdict(dict)
        # Rolling stats accumulators
        self._rolling: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.rolling_window))

        logger.info(
            "StreamProcessor initialized (window=%d, overlap=%.1f, fft_bins=%d)",
            self.window_size, self.overlap, self.fft_bins,
        )

    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a sensor event and return features when a window is complete."""
        asset_id = event.get("asset_id", "unknown")
        sensors = event.get("sensors", {})

        if not sensors:
            return None

        features = {}
        window_ready = False

        for sensor_name, value in sensors.items():
            # Ensure buffer exists
            if sensor_name not in self._buffers[asset_id]:
                self._buffers[asset_id][sensor_name] = WindowBuffer(
                    self.window_size, self.overlap
                )

            # Add to window buffer
            window = self._buffers[asset_id][sensor_name].add(value)

            # Add to rolling stats
            roll_key = f"{asset_id}:{sensor_name}"
            self._rolling[roll_key].append(value)

            # Compute rolling stats always
            roll_arr = np.array(self._rolling[roll_key])
            features[f"{sensor_name}_rolling_mean"] = float(np.mean(roll_arr))
            features[f"{sensor_name}_rolling_std"] = float(np.std(roll_arr))
            features[f"{sensor_name}_rolling_min"] = float(np.min(roll_arr))
            features[f"{sensor_name}_rolling_max"] = float(np.max(roll_arr))

            # If window is complete, compute windowed features
            if window is not None:
                window_ready = True
                # Time-domain features
                features[f"{sensor_name}_rms"] = float(np.sqrt(np.mean(window ** 2)))
                features[f"{sensor_name}_peak"] = float(np.max(np.abs(window)))
                features[f"{sensor_name}_crest_factor"] = (
                    features[f"{sensor_name}_peak"]
                    / max(features[f"{sensor_name}_rms"], 1e-10)
                )
                features[f"{sensor_name}_mean"] = float(np.mean(window))
                features[f"{sensor_name}_variance"] = float(np.var(window))

                std = np.std(window)
                if std > 1e-10:
                    normalized = (window - np.mean(window)) / std
                    features[f"{sensor_name}_skewness"] = float(np.mean(normalized ** 3))
                    features[f"{sensor_name}_kurtosis"] = float(np.mean(normalized ** 4))
                else:
                    features[f"{sensor_name}_skewness"] = 0.0
                    features[f"{sensor_name}_kurtosis"] = 0.0

                # FFT spectral features
                fft_features = self._compute_fft(window, sensor_name)
                features.update(fft_features)

        if not features:
            return None

        return {
            "asset_id": asset_id,
            "timestamp": event.get("timestamp"),
            "window_complete": window_ready,
            "features": features,
        }

    def _compute_fft(self, window: np.ndarray, sensor_name: str) -> Dict[str, float]:
        """Compute FFT-based spectral features."""
        n = len(window)

        # Apply Hanning window to reduce spectral leakage
        windowed = window * np.hanning(n)

        # Compute FFT
        fft_vals = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_vals)[:self.fft_bins]
        fft_power = fft_mag ** 2

        # Normalize
        total_power = np.sum(fft_power) + 1e-10

        features = {}

        # Dominant frequency bin
        features[f"{sensor_name}_fft_dominant_bin"] = float(np.argmax(fft_mag))
        features[f"{sensor_name}_fft_dominant_magnitude"] = float(np.max(fft_mag))

        # Spectral centroid
        freq_bins = np.arange(len(fft_mag))
        features[f"{sensor_name}_spectral_centroid"] = float(
            np.sum(freq_bins * fft_mag) / (np.sum(fft_mag) + 1e-10)
        )

        # Spectral entropy (uncertainty in frequency distribution)
        prob = fft_power / total_power
        prob = prob[prob > 0]
        features[f"{sensor_name}_spectral_entropy"] = float(-np.sum(prob * np.log2(prob)))

        # Band energies (low, mid, high thirds)
        third = len(fft_mag) // 3
        features[f"{sensor_name}_band_low"] = float(np.sum(fft_power[:third]))
        features[f"{sensor_name}_band_mid"] = float(np.sum(fft_power[third:2*third]))
        features[f"{sensor_name}_band_high"] = float(np.sum(fft_power[2*third:]))

        return features


class RustStreamProcessorAdapter:
    """Wraps Rust StreamProcessor with a Python-compatible interface."""

    def __init__(self, config: Dict[str, Any]):
        window_size = config.get("window_size_samples", 256)
        overlap = config.get("window_overlap", 0.5)
        fft_bins = config.get("fft_n_bins", 128)
        rolling_window = config.get("rolling_stats_window", 64)

        self._inner = RustStreamProcessor(
            window_size=window_size,
            overlap=overlap,
            fft_bins=fft_bins,
            rolling_window=rolling_window
        )
        logger.info("RustStreamProcessorAdapter initialized")

    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Bridge Python event dict to Rust process method."""
        asset_id = event.get("asset_id", "unknown")
        sensors = event.get("sensors", {})
        timestamp = event.get("timestamp", 0.0)

        # Rust process(asset_id, sensors_dict, timestamp)
        return self._inner.process(asset_id, sensors, timestamp)


def create_stream_processor(config: Dict[str, Any]) -> StreamProcessor:
    """Factory: returns Rust-accelerated or Python StreamProcessor.

    Prefers Rust when aurora_core is available.
    """
    if _HAS_RUST:
        logger.info("Creating Rust StreamProcessor")
        return RustStreamProcessorAdapter(config)

    logger.info("Creating Python StreamProcessor")
    return StreamProcessor(config)

