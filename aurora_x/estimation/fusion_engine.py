"""
AURORA-X Multi-Sensor Fusion Engine.

Fuses Kalman-filtered state estimates from multiple sensor channels
with covariance intersection, augmented by neural residual correction.
Produces structured observation vectors for the RL subsystem.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from aurora_x.estimation.kalman_filter import ExtendedKalmanFilter, create_kalman_filter
from aurora_x.estimation.neural_residual import NeuralResidualCorrector

logger = logging.getLogger("aurora_x.estimation.fusion")

# Sensor-to-state mapping
SENSOR_INDICES = {
    "thermal": 0,
    "vibration": 1,
    "pressure": 2,
    "flow": 3,
    "electrical": 4,
    "acoustic": 5,
}

STATE_DIM = 8  # temp, vib, pressure, flow, electrical, acoustic, degradation, trend
MEASUREMENT_DIM = 6  # 6 sensor channels


class AssetStateEstimator:
    """State estimator for a single industrial asset."""

    def __init__(self, asset_id: str, config: Dict[str, Any]):
        self.asset_id = asset_id
        self.config = config

        # Create Kalman filter
        self.kf = create_kalman_filter(config, STATE_DIM, MEASUREMENT_DIM)

        # Initialize state
        x0 = np.array([80.0, 1.0, 50.0, 100.0, 15.0, 40.0, 0.0, 0.0])
        self.kf.initialize(x0)

        # Setup state transition (simple first-order dynamics)
        self.kf.F = np.eye(STATE_DIM)
        self.kf.F[6, 6] = 1.0001  # Slow degradation growth
        self.kf.F[7, 6] = 0.01    # Trend follows degradation

        # Measurement matrix (sensors observe first 6 states directly)
        self.kf.H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        for i in range(MEASUREMENT_DIM):
            self.kf.H[i, i] = 1.0

        # Neural residual corrector
        self.corrector = NeuralResidualCorrector(config)

        # History for trend analysis
        self._state_history: List[np.ndarray] = []
        self._update_count = 0

    def update(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Update state estimate with new features.

        Returns:
            Dictionary with state estimates and uncertainties.
        """
        # Build measurement vector from features
        z = self._features_to_measurement(features)

        # Predict step
        state_pred, _ = self.kf.predict()

        # Neural correction
        state_corrected = self.corrector.correct(state_pred, z)

        # Update step with measurement
        state_updated, cov = self.kf.update(z)

        # Train the corrector
        self.corrector.observe(state_pred, z, state_updated)

        # Compute trend (rate of change)
        self._state_history.append(state_updated.copy())
        if len(self._state_history) > 100:
            self._state_history.pop(0)

        trend = self._compute_trend()
        state_updated[7] = trend  # Update trend state

        self._update_count += 1
        uncertainty = self.kf.uncertainty

        return {
            "state_vector": state_updated.tolist(),
            "covariance_diagonal": np.diag(cov).tolist(),
            "uncertainty": uncertainty.tolist(),
            "temperature": float(state_updated[0]),
            "vibration": float(state_updated[1]),
            "pressure": float(state_updated[2]),
            "flow": float(state_updated[3]),
            "electrical": float(state_updated[4]),
            "acoustic": float(state_updated[5]),
            "degradation": float(np.clip(state_updated[6], 0, 1)),
            "trend": float(trend),
            "confidence": float(1.0 / (1.0 + np.mean(uncertainty))),
            "update_count": self._update_count,
        }

    def _features_to_measurement(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to measurement vector."""
        feat = features.get("features", features)
        z = np.zeros(MEASUREMENT_DIM)

        # Map available sensor features to measurement vector
        for sensor, idx in SENSOR_INDICES.items():
            # Try different feature name patterns
            for pattern in [
                f"{sensor}_rolling_mean",
                f"{sensor}_rms",
                f"{sensor}_mean",
                sensor,
            ]:
                if pattern in feat:
                    z[idx] = feat[pattern]
                    break

        return z

    def _compute_trend(self) -> float:
        """Compute degradation trend from state history."""
        if len(self._state_history) < 10:
            return 0.0

        # Use degradation state (index 6) over recent history
        degradation_vals = [s[6] for s in self._state_history[-50:]]
        if len(degradation_vals) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(degradation_vals))
        coeffs = np.polyfit(x, degradation_vals, 1)
        return float(coeffs[0])  # Slope


class FusionEngine:
    """Multi-asset state estimation and fusion engine."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._estimators: Dict[str, AssetStateEstimator] = {}
        logger.info("FusionEngine initialized")

    def _get_estimator(self, asset_id: str) -> AssetStateEstimator:
        """Get or create an estimator for an asset."""
        if asset_id not in self._estimators:
            self._estimators[asset_id] = AssetStateEstimator(asset_id, self.config)
            logger.info("Created state estimator for asset: %s", asset_id)
        return self._estimators[asset_id]

    def update(self, asset_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Update state estimate for an asset with new features."""
        estimator = self._get_estimator(asset_id)
        return estimator.update(features)

    def get_state(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get current state estimate for an asset."""
        if asset_id not in self._estimators:
            return None
        est = self._estimators[asset_id]
        return {
            "state_vector": est.kf.state.tolist(),
            "uncertainty": est.kf.uncertainty.tolist(),
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state estimates for all tracked assets."""
        return {
            asset_id: self.get_state(asset_id)
            for asset_id in self._estimators
        }

    @property
    def tracked_assets(self) -> List[str]:
        return list(self._estimators.keys())
