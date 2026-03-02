"""
AURORA-X Spectral Analysis Module.

FFT-based spectral decomposition, envelope analysis,
and order tracking for rotational machinery diagnostics.

When aurora_core is available, uses Rust-backed rustfft for faster FFT.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("aurora_x.pipeline.spectral")

# ── Try Rust acceleration ──
try:
    from aurora_core import SpectralAnalyzer as RustSpectralAnalyzer
    _HAS_RUST = True
    logger.info("Rust-accelerated SpectralAnalyzer available (aurora_core)")
except ImportError:
    _HAS_RUST = False


class SpectralAnalyzer:
    """Advanced spectral analysis for vibration and acoustic signals."""

    def __init__(self, sample_rate: float = 100.0, n_fft: int = 256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freq_resolution = sample_rate / n_fft

    def full_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute full frequency spectrum.

        Returns:
            (frequencies, magnitudes) arrays.
        """
        n = len(signal)
        windowed = signal * np.hanning(n)
        fft_vals = np.fft.rfft(windowed, n=self.n_fft)
        magnitudes = 2.0 * np.abs(fft_vals) / n
        frequencies = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        return frequencies, magnitudes

    def power_spectral_density(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density using Welch's method approximation."""
        frequencies, magnitudes = self.full_spectrum(signal)
        psd = magnitudes ** 2 / self.freq_resolution
        return frequencies, psd

    def envelope_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Envelope analysis via Hilbert transform for bearing fault detection."""
        # Analytic signal via FFT-based Hilbert transform
        n = len(signal)
        fft_signal = np.fft.fft(signal)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2

        analytic = np.fft.ifft(fft_signal * h)
        envelope = np.abs(analytic)

        # Spectrum of envelope
        return self.full_spectrum(envelope - np.mean(envelope))

    def order_analysis(
        self, signal: np.ndarray, shaft_freq: float
    ) -> Dict[str, float]:
        """Compute order-based amplitudes relative to shaft frequency."""
        frequencies, magnitudes = self.full_spectrum(signal)

        orders = {}
        for order in [1, 2, 3, 4, 5, 6]:
            target_freq = order * shaft_freq
            idx = np.argmin(np.abs(frequencies - target_freq))
            orders[f"order_{order}x"] = float(magnitudes[idx])

        return orders

    def bearing_frequencies(
        self,
        shaft_freq: float,
        n_balls: int = 8,
        ball_dia: float = 10.0,
        pitch_dia: float = 45.0,
        contact_angle: float = 0.0,
    ) -> Dict[str, float]:
        """Calculate characteristic bearing defect frequencies."""
        cos_angle = np.cos(np.radians(contact_angle))
        ratio = ball_dia / pitch_dia

        bpfo = (n_balls / 2) * shaft_freq * (1 - ratio * cos_angle)
        bpfi = (n_balls / 2) * shaft_freq * (1 + ratio * cos_angle)
        bsf = (pitch_dia / (2 * ball_dia)) * shaft_freq * (
            1 - (ratio * cos_angle) ** 2
        )
        ftf = (shaft_freq / 2) * (1 - ratio * cos_angle)

        return {
            "BPFO": bpfo,  # Ball Pass Freq Outer Race
            "BPFI": bpfi,  # Ball Pass Freq Inner Race
            "BSF": bsf,    # Ball Spin Frequency
            "FTF": ftf,    # Fundamental Train Frequency
        }

    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive spectral features from a signal window."""
        frequencies, magnitudes = self.full_spectrum(signal)
        power = magnitudes ** 2
        total_power = np.sum(power) + 1e-10

        features = {}

        # Peak frequency
        peak_idx = np.argmax(magnitudes)
        features["peak_frequency"] = float(frequencies[peak_idx])
        features["peak_magnitude"] = float(magnitudes[peak_idx])

        # Spectral centroid
        features["spectral_centroid"] = float(
            np.sum(frequencies * magnitudes) / (np.sum(magnitudes) + 1e-10)
        )

        # Spectral spread (bandwidth)
        centroid = features["spectral_centroid"]
        features["spectral_spread"] = float(
            np.sqrt(
                np.sum(((frequencies - centroid) ** 2) * magnitudes)
                / (np.sum(magnitudes) + 1e-10)
            )
        )

        # Spectral entropy
        prob = power / total_power
        prob = prob[prob > 0]
        features["spectral_entropy"] = float(-np.sum(prob * np.log2(prob)))

        # Spectral flatness (tonality)
        geo_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
        arith_mean = np.mean(magnitudes)
        features["spectral_flatness"] = float(geo_mean / (arith_mean + 1e-10))

        # Spectral rolloff (95%)
        cumsum = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumsum, 0.95 * total_power)
        features["spectral_rolloff"] = float(
            frequencies[min(rolloff_idx, len(frequencies) - 1)]
        )

        return features


def create_spectral_analyzer(sample_rate: float = 100.0, n_fft: int = 256):
    """Factory: returns Rust or Python SpectralAnalyzer.

    Rust version has identical method signatures — direct substitution.
    """
    if _HAS_RUST:
        logger.info("Creating Rust SpectralAnalyzer (sr=%.0f, nfft=%d)", sample_rate, n_fft)
        return RustSpectralAnalyzer(sample_rate=sample_rate, n_fft=n_fft)
    logger.info("Creating Python SpectralAnalyzer (sr=%.0f, nfft=%d)", sample_rate, n_fft)
    return SpectralAnalyzer(sample_rate, n_fft)

