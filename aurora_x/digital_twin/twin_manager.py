"""
AURORA-X Digital Twin Manager.

Manages real-time synchronization (shadow mode) and deterministic simulation
(training mode) for physics-informed digital twins.
"""

import numpy as np
import logging
import copy
from typing import Dict, Any, Optional, List

from aurora_x.digital_twin.physics_engine import PhysicsEngine, create_physics_engine
from aurora_x.digital_twin.degradation_model import (
    WeibullDegradationModel, BayesianRULEstimator,
    create_weibull_model, create_bayesian_rul,
)

logger = logging.getLogger("aurora_x.digital_twin.manager")


class DigitalTwin:
    """Digital twin instance for a single industrial asset."""

    def __init__(self, asset_id: str, config: Dict[str, Any]):
        self.asset_id = asset_id
        self.config = config

        # Physics engine (Rust-accelerated when available)
        self.physics = create_physics_engine(config)
        self.physics.create_asset(asset_id)

        # Degradation model (Rust-accelerated when available)
        weibull_shape = config.get("degradation", {}).get("weibull_shape", 2.5)
        weibull_scale = config.get("degradation", {}).get("weibull_scale", 10000)
        self.weibull = create_weibull_model(shape=weibull_shape, scale=weibull_scale)

        # Bayesian RUL estimator (Rust-accelerated when available)
        self.rul_estimator = create_bayesian_rul(config.get("degradation", {}))

        # Shared physics for simulations
        self._sim_physics = PhysicsEngine(self.config)
        self._sim_physics.create_asset("sim")

        # Twin state
        self.operating_time = 0.0
        self.total_cycles = 0
        self._synced_state: Optional[Dict] = None
        self._last_action: Optional[np.ndarray] = None


    def update(self, state: Dict[str, Any]):
        """Synchronize twin with real sensor state (shadow mode)."""
        self._synced_state = state

        # Extract degradation from state
        degradation = state.get("degradation", 0.0)
        self.operating_time += 1.0

        # Update Bayesian RUL with observation (Decoupled: only every 10 cycles to save CPU)
        if self.total_cycles % 10 == 0:
            self.rul_estimator.observe(self.operating_time, degradation)

        # Step physics engine
        self.physics.step(self.asset_id)
        self.total_cycles += 1

    def simulate_action(self, action: np.ndarray, horizon: int = 100) -> Dict[str, Any]:
        """Simulate the effect of an action over a time horizon (counterfactual).
        
        This uses a cached simulation engine and resets it for each call.
        """
        # Save current state
        current_physics_state = self.physics.get_state(self.asset_id)
        if current_physics_state is None:
            return {"error": "No physics state available"}

        # Reset simulation state to current matching state
        self._sim_physics._assets["sim"]["state"] = current_physics_state.copy()
        self._sim_physics._assets["sim"]["time"] = self.operating_time

        # Simulate forward
        trajectory = []
        for step in range(horizon):
            result = self._sim_physics.step("sim", action)
            trajectory.append(result)

        final_state = trajectory[-1]
        return {
            "trajectory": trajectory,
            "final_bearing_degradation": final_state["bearing_degradation"],
            "final_bearing_temp": final_state["bearing_temp"],
            "final_seal_degradation": final_state["seal_degradation"],
            "horizon_steps": horizon,
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report for this asset."""
        physics_state = self.physics.get_state(self.asset_id)
        rul = self.rul_estimator.estimate_rul(self.operating_time)
        
        # Compute health score (inverse of degradation)
        degradation = self._synced_state.get("degradation", 0.0) if self._synced_state else 0.0
        health_score = (1.0 - degradation) * 100.0
        
        # Weibull metrics
        hazard = self.weibull.hazard_rate(self.operating_time)
        survival = self.weibull.survival_probability(self.operating_time)

        # Compute Trends (Past, Present, Future)
        # We sample 10 points for a smooth sparkline/bar trend
        trend = []
        
        # 1. Past (from observations)
        obs = self.rul_estimator._observations if hasattr(self.rul_estimator, "_observations") else []
        if obs:
            # Sample 4 historical points
            indices = np.linspace(0, len(obs) - 1, 4, dtype=int)
            for idx in indices:
                trend.append(float(obs[idx][1]))
        else:
            trend.extend([0.0] * 4)

        # 2. Present
        trend.append(float(degradation))

        # 3. Future Projection (predictive)
        # Project forward by 25%, 50%, 75%, 100% of the current RUL mean
        projected_rul = rul.get("rul_mean", 1000)
        for pct in [0.25, 0.5, 0.75, 1.0, 1.25]:
            future_t = self.operating_time + (projected_rul * pct)
            future_deg = self.weibull.failure_probability(future_t)
            trend.append(float(future_deg))

        return {
            "asset_id": self.asset_id,
            "operating_time": self.operating_time,
            "total_cycles": self.total_cycles,
            "health_score": health_score,
            "degradation": degradation,
            "hazard_rate": hazard,
            "survival_probability": survival,
            "failure_probability": 1.0 - survival,
            "rul": rul,
            "degradation_trend": trend,
            "physics_state": physics_state.tolist() if physics_state is not None else None,
            "synced_state": self._synced_state,
        }


class TwinManager:
    """Manages digital twins for all industrial assets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get("mode", "realtime")  # realtime | simulation
        self._twins: Dict[str, DigitalTwin] = {}

        logger.info("TwinManager initialized (mode=%s)", self.mode)

    def _get_twin(self, asset_id: str) -> DigitalTwin:
        """Get or create a digital twin for an asset."""
        if asset_id not in self._twins:
            self._twins[asset_id] = DigitalTwin(asset_id, self.config)
            logger.info("Digital twin created for asset: %s", asset_id)
        return self._twins[asset_id]

    def update(self, asset_id: str, state: Dict[str, Any]):
        """Update a twin with new sensor-fused state."""
        twin = self._get_twin(asset_id)
        twin.update(state)

    def get_twin(self, asset_id: str) -> Optional[DigitalTwin]:
        """Get a twin instance."""
        return self._twins.get(asset_id)

    def get_health_report(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get health report for an asset."""
        twin = self._twins.get(asset_id)
        if twin is None:
            return None
        return twin.get_health_report()

    def simulate_action(
        self, asset_id: str, action: np.ndarray, horizon: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Counterfactual simulation for RL training."""
        twin = self._twins.get(asset_id)
        if twin is None:
            return None
        return twin.simulate_action(action, horizon)

    def get_all_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get health reports for all assets."""
        return {
            asset_id: twin.get_health_report()
            for asset_id, twin in self._twins.items()
        }

    @property
    def asset_ids(self) -> List[str]:
        return list(self._twins.keys())

    def reset(self, asset_id: Optional[str] = None):
        """Reset twin(s) to initial state."""
        if asset_id:
            if asset_id in self._twins:
                del self._twins[asset_id]
        else:
            self._twins.clear()
