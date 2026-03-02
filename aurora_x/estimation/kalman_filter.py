"""
AURORA-X Kalman Filter Implementations.

Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
for multi-sensor state estimation with uncertainty propagation.

When aurora_core (Rust/PyO3) is available, native EKF/UKF can be
imported directly for 10-100x faster matrix operations.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
import logging

logger = logging.getLogger("aurora_x.estimation.kalman")

# ── Try Rust acceleration ──
try:
    from aurora_core import ExtendedKalmanFilter as RustEKF
    from aurora_core import UnscentedKalmanFilter as RustUKF
    _HAS_RUST = True
    logger.info("Rust-accelerated Kalman filters available (aurora_core)")
except ImportError:
    _HAS_RUST = False


# ── Rust Adapters (bridge Rust method-based API to Python property-based API) ──

class RustEKFAdapter:
    """Wraps Rust EKF with Python-compatible interface."""

    def __init__(self, state_dim, measurement_dim, process_noise=0.01, measurement_noise=0.05):
        self.n = state_dim
        self.m = measurement_dim
        self._inner = RustEKF(state_dim, measurement_dim)

        # Mutable Jacobians (Python code sets these directly)
        self.F = np.eye(state_dim)
        self.H = np.eye(measurement_dim, state_dim)
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(measurement_dim) * measurement_noise
        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self._inner.initialize(x0, P0)
        self._initialized = True

    def predict(self, f=None, F_jacobian=None, dt=1.0):
        F = F_jacobian if F_jacobian is not None else self.F
        self._inner.predict(F, dt)
        return self.state, self.covariance

    def update(self, z, h=None, H_jacobian=None):
        H = H_jacobian if H_jacobian is not None else self.H
        self._inner.update(z, H)
        return self.state, self.covariance

    @property
    def state(self):
        return np.array(self._inner.state())

    @property
    def covariance(self):
        return np.array(self._inner.covariance())

    @property
    def uncertainty(self):
        return np.array(self._inner.uncertainty())


class RustUKFAdapter:
    """Wraps Rust UKF with Python-compatible interface."""

    def __init__(self, state_dim, measurement_dim, process_noise=0.01,
                 measurement_noise=0.05, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = state_dim
        self.m = measurement_dim
        self._inner = RustUKF(state_dim, measurement_dim)
        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self._inner.initialize(x0, P0)
        self._initialized = True

    def predict(self, f=None, dt=1.0):
        self._inner.predict(dt)
        return self.state, self.covariance

    def update(self, z, h=None):
        self._inner.update(z)
        return self.state, self.covariance

    @property
    def state(self):
        return np.array(self._inner.state())

    @property
    def covariance(self):
        return np.array(self._inner.covariance())

    @property
    def uncertainty(self):
        return np.array(self._inner.uncertainty())


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state estimation.

    State vector: [temperature, vibration_rms, pressure, flow, degradation, ...]
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
    ):
        self.n = state_dim
        self.m = measurement_dim

        # State estimate and covariance
        self.x = np.zeros(state_dim)       # State mean
        self.P = np.eye(state_dim) * 1.0   # State covariance

        # Noise covariances
        self.Q = np.eye(state_dim) * process_noise         # Process noise
        self.R = np.eye(measurement_dim) * measurement_noise  # Measurement noise

        # Jacobians (default to linear identity)
        self.F = np.eye(state_dim)         # State transition Jacobian
        self.H = np.eye(measurement_dim, state_dim)  # Measurement Jacobian

        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        """Initialize filter state."""
        self.x = x0.copy()
        if P0 is not None:
            self.P = P0.copy()
        self._initialized = True

    def predict(
        self,
        f: Optional[Callable] = None,
        F_jacobian: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step."""
        if f is not None:
            self.x = f(self.x, dt)
        else:
            self.x = self.F @ self.x

        F = F_jacobian if F_jacobian is not None else self.F
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy(), self.P.copy()

    def update(
        self,
        z: np.ndarray,
        h: Optional[Callable] = None,
        H_jacobian: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement update step."""
        H = H_jacobian if H_jacobian is not None else self.H

        if h is not None:
            z_pred = h(self.x)
        else:
            z_pred = H @ self.x

        # Innovation
        y = z - z_pred
        S = H @ self.P @ H.T + self.R  # Innovation covariance

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T  # Joseph form

        return self.x.copy(), self.P.copy()

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def uncertainty(self) -> np.ndarray:
        """Standard deviation of each state variable."""
        return np.sqrt(np.diag(self.P))


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for highly nonlinear systems.

    Uses sigma point propagation instead of Jacobian linearization.
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.n = state_dim
        self.m = measurement_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1.0
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(measurement_dim) * measurement_noise

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha ** 2 * (state_dim + kappa) - state_dim

        # Sigma point weights
        self.n_sigma = 2 * state_dim + 1
        self.Wm = np.zeros(self.n_sigma)  # Mean weights
        self.Wc = np.zeros(self.n_sigma)  # Covariance weights

        self.Wm[0] = self.lambda_ / (state_dim + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha ** 2 + beta)
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2 * (state_dim + self.lambda_))
            self.Wc[i] = self.Wm[i]

        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        self.x = x0.copy()
        if P0 is not None:
            self.P = P0.copy()
        self._initialized = True

    def _sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points."""
        n = self.n
        sigma = np.zeros((self.n_sigma, n))
        sigma[0] = self.x

        sqrt_P = np.linalg.cholesky((n + self.lambda_) * self.P)

        for i in range(n):
            sigma[i + 1] = self.x + sqrt_P[i]
            sigma[n + i + 1] = self.x - sqrt_P[i]

        return sigma

    def predict(
        self, f: Callable, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step with sigma point propagation."""
        sigma = self._sigma_points()

        # Propagate sigma points through dynamics
        sigma_pred = np.array([f(s, dt) for s in sigma])

        # Weighted mean
        self.x = np.sum(self.Wm[:, None] * sigma_pred, axis=0)

        # Weighted covariance
        self.P = self.Q.copy()
        for i in range(self.n_sigma):
            diff = sigma_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

        self._sigma_pred = sigma_pred
        return self.x.copy(), self.P.copy()

    def update(
        self, z: np.ndarray, h: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Measurement update with sigma point propagation."""
        sigma = self._sigma_points()

        # Propagate through measurement model
        z_sigma = np.array([h(s) for s in sigma])

        # Predicted measurement
        z_pred = np.sum(self.Wm[:, None] * z_sigma, axis=0)

        # Innovation covariance
        S = self.R.copy()
        Pxz = np.zeros((self.n, self.m))
        for i in range(self.n_sigma):
            dz = z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)
            dx = sigma[i] - self.x
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T

        return self.x.copy(), self.P.copy()

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def uncertainty(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P))


def create_kalman_filter(config: Dict[str, Any], state_dim: int, measurement_dim: int):
    """Factory to create the right KF variant based on config.

    Prefers Rust-accelerated implementation when aurora_core is available.
    """
    kf_type = config.get("kalman", {}).get("type", "ekf")
    process_noise = config.get("kalman", {}).get("process_noise", 0.01)
    measurement_noise = config.get("kalman", {}).get("measurement_noise", 0.05)

    if _HAS_RUST:
        if kf_type == "ukf":
            logger.info("Creating Rust UKF (state_dim=%d, meas_dim=%d)", state_dim, measurement_dim)
            return RustUKFAdapter(state_dim, measurement_dim, process_noise, measurement_noise)
        logger.info("Creating Rust EKF (state_dim=%d, meas_dim=%d)", state_dim, measurement_dim)
        return RustEKFAdapter(state_dim, measurement_dim, process_noise, measurement_noise)

    if kf_type == "ukf":
        logger.info("Creating Python UKF (state_dim=%d, meas_dim=%d)", state_dim, measurement_dim)
        return UnscentedKalmanFilter(state_dim, measurement_dim, process_noise, measurement_noise)
    logger.info("Creating Python EKF (state_dim=%d, meas_dim=%d)", state_dim, measurement_dim)
    return ExtendedKalmanFilter(state_dim, measurement_dim, process_noise, measurement_noise)

