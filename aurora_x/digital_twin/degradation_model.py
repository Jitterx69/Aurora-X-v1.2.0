"""
AURORA-X Degradation Modeling.

Weibull hazard functions and Bayesian Remaining Useful Life (RUL) inference
for predictive maintenance and risk assessment.

When aurora_core is available, uses Rust-accelerated Weibull and BayesianRUL.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats as scipy_stats

logger = logging.getLogger("aurora_x.digital_twin.degradation")

# ── Try Rust acceleration ──
try:
    from aurora_core import WeibullModel as _RustWeibull
    from aurora_core import BayesianRUL as _RustBayesianRUL
    _HAS_RUST = True
    logger.info("Rust-accelerated degradation models available (aurora_core)")
except ImportError:
    _HAS_RUST = False


class RustWeibullAdapter:
    """Wraps Rust WeibullModel with WeibullDegradationModel interface."""

    def __init__(self, shape: float = 2.5, scale: float = 10000):
        self.beta = shape
        self.eta = scale
        self._inner = _RustWeibull(shape=shape, scale=scale)

    def hazard_rate(self, t: float) -> float:
        return self._inner.hazard_rate(t)

    def survival_probability(self, t: float) -> float:
        return self._inner.survival_probability(t)

    def failure_probability(self, t: float) -> float:
        return self._inner.cdf(t)

    def pdf(self, t: float) -> float:
        return self._inner.pdf(t)

    def mean_life(self) -> float:
        return self._inner.mttf()

    def remaining_useful_life(self, current_age: float, confidence: float = 0.9) -> float:
        # Rust doesn't have conditional RUL — compute analytically
        target_p = 1.0 - confidence
        surv = self.survival_probability(current_age)
        if surv < 1e-10:
            return 0.0
        t_target = self.eta * (-np.log(target_p * surv)) ** (1 / self.beta)
        return max(0, t_target - current_age)


class RustBayesianRULAdapter:
    """Wraps Rust BayesianRUL with BayesianRULEstimator interface."""

    def __init__(self, config: Dict[str, Any]):
        self._inner = _RustBayesianRUL()
        self._observations: List[Tuple[float, float]] = []

    def observe(self, time: float, degradation: float):
        self._inner.observe(time, degradation)
        self._observations.append((time, degradation))

    def estimate_rul(self, current_time: float) -> Dict[str, float]:
        result = self._inner.estimate_rul(current_time)
        # Map Rust keys to Python keys
        return {
            "rul_mean": result.get("mean", 0.0),
            "rul_std": result.get("std", 0.0),
            "rul_median": result.get("median", 0.0),
            "rul_p10": result.get("p10", 0.0),
            "rul_p25": result.get("p25", 0.0),
            "rul_p75": result.get("p75", 0.0),
            "rul_p90": result.get("p90", 0.0),
            "confidence_interval_90": (
                result.get("ci_lower", 0.0),
                result.get("ci_upper", 0.0),
            ),
            "n_observations": len(self._observations),
        }

class WeibullDegradationModel:
    """Weibull distribution-based degradation and failure modeling.

    The Weibull hazard function models increasing failure probability:
        h(t) = (beta / eta) * (t / eta)^(beta - 1)
    Where:
        beta = shape parameter (>1 means wear-out failure)
        eta  = scale parameter (characteristic life)
    """

    def __init__(self, shape: float = 2.5, scale: float = 10000):
        self.beta = shape   # Shape parameter
        self.eta = scale    # Scale parameter (cycles to characteristic failure)

    def hazard_rate(self, t: float) -> float:
        """Instantaneous hazard rate at time t."""
        if t <= 0:
            return 0.0
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1)

    def survival_probability(self, t: float) -> float:
        """Probability of surviving beyond time t."""
        if t <= 0:
            return 1.0
        return np.exp(-(t / self.eta) ** self.beta)

    def failure_probability(self, t: float) -> float:
        """Cumulative probability of failure by time t."""
        return 1.0 - self.survival_probability(t)

    def pdf(self, t: float) -> float:
        """Probability density function at time t."""
        if t <= 0:
            return 0.0
        return self.hazard_rate(t) * self.survival_probability(t)

    def mean_life(self) -> float:
        """Expected (mean) lifetime."""
        from scipy.special import gamma
        return self.eta * gamma(1 + 1 / self.beta)

    def remaining_useful_life(self, current_age: float, confidence: float = 0.9) -> float:
        """Point estimate of RUL at given confidence level.

        Returns the time to reach the given failure probability.
        """
        target_p = 1.0 - confidence
        # Solve: survival(current_age + RUL) / survival(current_age) = target_p
        # conditional survival
        if self.survival_probability(current_age) < 1e-10:
            return 0.0

        # Quantile of conditional distribution
        t_target = self.eta * (-np.log(target_p * self.survival_probability(current_age))) ** (1 / self.beta)
        rul = max(0, t_target - current_age)
        return rul


class BayesianRULEstimator:
    """Bayesian Remaining Useful Life estimation.

    Uses particle-based inference to estimate RUL distribution
    given observed degradation trajectory.
    """

    def __init__(self, config: Dict[str, Any]):
        self.n_particles = config.get("bayesian_samples", 100)
        self.failure_threshold = config.get("failure_threshold", 0.8)

        # Prior on Weibull parameters
        self._shape_samples = np.random.lognormal(mean=np.log(2.5), sigma=0.3, size=self.n_particles)
        self._scale_samples = np.random.lognormal(mean=np.log(10000), sigma=0.5, size=self.n_particles)

        # Weights (uniform prior)
        self._weights = np.ones(self.n_particles) / self.n_particles

        self._observations: List[Tuple[float, float]] = []  # (time, degradation)

        logger.info("BayesianRULEstimator initialized (particles=%d)", self.n_particles)

    def observe(self, time: float, degradation: float):
        """Add a degradation observation."""
        self._observations.append((time, degradation))

        # Likelihood-based weight update
        if len(self._observations) >= 2:
            self._update_weights(time, degradation)

    def _update_weights(self, t: float, d: float):
        """Update particle weights based on observation likelihood."""
        likelihoods = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            model = WeibullDegradationModel(
                shape=self._shape_samples[i],
                scale=self._scale_samples[i],
            )
            # Expected degradation at time t
            expected_d = model.failure_probability(t)
            # Gaussian likelihood
            sigma = 0.05
            likelihoods[i] = np.exp(-0.5 * ((d - expected_d) / sigma) ** 2)

        # Normalize weights
        total = np.sum(self._weights * likelihoods)
        if total > 1e-10:
            self._weights = (self._weights * likelihoods) / total
        else:
            # Reset if all weights collapse
            self._weights = np.ones(self.n_particles) / self.n_particles

        # Effective sample size check - resample if needed
        ess = 1.0 / np.sum(self._weights ** 2)
        if ess < self.n_particles / 2:
            self._resample()

    def _resample(self):
        """Systematic resampling of particles."""
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0  # Ensure sum is exactly 1

        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.searchsorted(cumsum, positions)

        self._shape_samples = self._shape_samples[indices]
        self._scale_samples = self._scale_samples[indices]
        self._weights = np.ones(self.n_particles) / self.n_particles

        # Add small perturbation (jitter) to avoid particle collapse
        self._shape_samples += np.random.normal(0, 0.05, self.n_particles)
        self._scale_samples += np.random.normal(0, 50, self.n_particles)
        self._shape_samples = np.clip(self._shape_samples, 1.0, 10.0)
        self._scale_samples = np.clip(self._scale_samples, 100, 100000)

    def estimate_rul(self, current_time: float) -> Dict[str, float]:
        """Estimate RUL distribution.

        Returns:
            Dict with RUL statistics: mean, median, std, percentiles, confidence interval.
        """
        rul_samples = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            model = WeibullDegradationModel(
                shape=self._shape_samples[i],
                scale=self._scale_samples[i],
            )
            rul_samples[i] = model.remaining_useful_life(current_time, confidence=0.9)

        # Weighted statistics
        rul_mean = np.average(rul_samples, weights=self._weights)
        rul_var = np.average((rul_samples - rul_mean) ** 2, weights=self._weights)

        sorted_idx = np.argsort(rul_samples)
        sorted_rul = rul_samples[sorted_idx]
        sorted_weights = self._weights[sorted_idx]
        cumulative = np.cumsum(sorted_weights)

        def weighted_percentile(p):
            idx = np.searchsorted(cumulative, p)
            idx = min(idx, len(sorted_rul) - 1)
            return float(sorted_rul[idx])

        return {
            "rul_mean": float(rul_mean),
            "rul_std": float(np.sqrt(rul_var)),
            "rul_median": weighted_percentile(0.5),
            "rul_p10": weighted_percentile(0.1),
            "rul_p25": weighted_percentile(0.25),
            "rul_p75": weighted_percentile(0.75),
            "rul_p90": weighted_percentile(0.9),
            "confidence_interval_90": (weighted_percentile(0.05), weighted_percentile(0.95)),
            "n_observations": len(self._observations),
        }


def create_weibull_model(shape: float = 2.5, scale: float = 10000):
    """Factory: returns Rust or Python Weibull model."""
    if _HAS_RUST:
        logger.debug("Creating Rust WeibullModel")
        return RustWeibullAdapter(shape, scale)
    return WeibullDegradationModel(shape, scale)


def create_bayesian_rul(config: Dict[str, Any]):
    """Factory: returns Rust or Python BayesianRUL."""
    if _HAS_RUST:
        logger.debug("Creating Rust BayesianRUL")
        return RustBayesianRULAdapter(config)
    return BayesianRULEstimator(config)

