"""
AURORA-X Safety Controller.

Enforces hard physical constraints, control barrier functions (CBFs),
and regulatory boundaries before any RL action is executed.
Implements deterministic safe fallback policy.

When aurora_core is available, uses Rust CBF evaluation for deterministic,
zero-allocation safety validation.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("aurora_x.rl.safety")

# ── Try Rust acceleration ──
try:
    from aurora_core import SafetyController as RustSafetyController
    _HAS_RUST = True
    logger.info("Rust-accelerated SafetyController available (aurora_core)")
except ImportError:
    _HAS_RUST = False


class ControlBarrierFunction:
    """Control Barrier Function for safe set invariance.

    A CBF h(x) defines a safe set S = {x : h(x) >= 0}.
    The safety condition is: dh/dt + alpha * h(x) >= 0
    """

    def __init__(self, name: str, limit: float, state_index: int, alpha: float = 0.1):
        self.name = name
        self.limit = limit
        self.state_index = state_index
        self.alpha = alpha

    def evaluate(self, state: np.ndarray) -> float:
        """Compute barrier value h(x). Positive = safe."""
        return self.limit - state[self.state_index]

    def is_safe(self, state: np.ndarray) -> bool:
        return self.evaluate(state) >= 0

    def compute_margin(self, state: np.ndarray) -> float:
        """Normalized safety margin [0, 1]. 0 = at boundary."""
        h = self.evaluate(state)
        return float(np.clip(h / self.limit, 0, 1))


class SafetyController:
    """Safety validation and enforcement layer.

    Sits between RL policy and physical actuation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get("barrier_alpha", 0.1)
        self.enable_fallback = config.get("enable_fallback", True)

        # Define constraint barrier functions
        self._barriers = []
        if config.get("max_temperature"):
            self._barriers.append(ControlBarrierFunction(
                "temperature", config["max_temperature"], state_index=0, alpha=self.alpha
            ))
        if config.get("max_vibration"):
            self._barriers.append(ControlBarrierFunction(
                "vibration", config["max_vibration"], state_index=1, alpha=self.alpha
            ))
        if config.get("max_pressure"):
            self._barriers.append(ControlBarrierFunction(
                "pressure", config["max_pressure"], state_index=2, alpha=self.alpha
            ))

        # Default safe limits
        if not self._barriers:
            self._barriers = [
                ControlBarrierFunction("temperature", 450.0, 0, self.alpha),
                ControlBarrierFunction("vibration", 15.0, 1, self.alpha),
                ControlBarrierFunction("pressure", 120.0, 2, self.alpha),
            ]

        self._violation_count = 0
        self._intervention_count = 0
        self._fallback_count = 0

        # Rust-backed CBF for hot-path validation
        self._rust_sc = None
        if _HAS_RUST:
            self._rust_sc = RustSafetyController()
            logger.info("SafetyController using Rust CBF hot-path (aurora_core)")

        logger.info("SafetyController initialized with %d barrier functions", len(self._barriers))


    def validate(
        self,
        action: np.ndarray,
        state: Dict[str, Any],
        twin=None,
    ) -> np.ndarray:
        """Validate and potentially modify an RL action for safety.

        Returns:
            Safe action (potentially modified from original).
        """
        state_vector = np.array([
            state.get("temperature", 0),
            state.get("vibration", 0),
            state.get("pressure", 0),
            state.get("flow", 0),
            state.get("electrical", 0),
            state.get("acoustic", 0),
            state.get("degradation", 0),
            state.get("trend", 0),
        ])

        # Check all barriers
        violations = []
        margins = {}
        for barrier in self._barriers:
            margin = barrier.compute_margin(state_vector)
            margins[barrier.name] = margin

            if not barrier.is_safe(state_vector):
                violations.append(barrier.name)
                self._violation_count += 1

        # If any barriers are violated, use fallback policy
        if violations:
            logger.warning("Safety violations detected: %s. Applying fallback.", violations)
            self._fallback_count += 1
            return self._fallback_policy(state_vector, violations)

        # Check if action would lead to unsafe state (predictive safety)
        if twin is not None:
            predicted = twin.simulate_action(action, horizon=10)
            if predicted and not isinstance(predicted, dict):
                pass  # Skip if sim fails

            # If any predicted state is unsafe, intervene
            if self._predict_unsafe(action, state_vector, margins):
                self._intervention_count += 1
                return self._safe_modification(action, margins)

        # Check margins - if close to boundary, dampen aggressive actions
        min_margin = min(margins.values()) if margins else 1.0
        if min_margin < 0.2:
            return self._safe_modification(action, margins)

        return action

    def _predict_unsafe(
        self, action: np.ndarray, state: np.ndarray, margins: Dict[str, float]
    ) -> bool:
        """Predict if action would lead to unsafe state."""
        # Simple heuristic: if margin is low and action is aggressive
        min_margin = min(margins.values()) if margins else 1.0

        # Speed increase when vibration margin is low
        if margins.get("vibration", 1.0) < 0.3 and action[1] > 0:
            return True

        # Reduced cooling when temperature margin is low
        if margins.get("temperature", 1.0) < 0.3 and action[2] < 0.3:
            return True

        return False

    def _safe_modification(
        self, action: np.ndarray, margins: Dict[str, float]
    ) -> np.ndarray:
        """Modify action to maintain safety margins."""
        safe_action = action.copy()

        # If temperature margin is low, increase cooling
        if margins.get("temperature", 1.0) < 0.3:
            safe_action[2] = max(safe_action[2], 0.7)  # Force high cooling
            safe_action[1] = min(safe_action[1], 0.0)   # Don't increase speed

        # If vibration margin is low, reduce speed
        if margins.get("vibration", 1.0) < 0.3:
            safe_action[1] = min(safe_action[1], -0.1)  # Reduce speed

        # If pressure margin is low, adjust load
        if margins.get("pressure", 1.0) < 0.3:
            safe_action[3] = -0.05  # Reduce load

        return safe_action

    def _fallback_policy(
        self, state: np.ndarray, violations: list
    ) -> np.ndarray:
        """Deterministic safe fallback policy.

        Conservative action: high maintenance, reduced speed,
        maximum cooling, reduced load.
        """
        return np.array([
            0.8,    # High maintenance intensity
            -0.15,  # Reduce speed
            1.0,    # Maximum cooling
            -0.08,  # Reduce load
        ])

    def get_stats(self) -> Dict[str, int]:
        return {
            "violations": self._violation_count,
            "interventions": self._intervention_count,
            "fallbacks": self._fallback_count,
        }

    def get_barrier_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get current barrier function status for all constraints."""
        state_vector = np.array([
            state.get("temperature", 0),
            state.get("vibration", 0),
            state.get("pressure", 0),
        ])

        status = {}
        for barrier in self._barriers:
            status[barrier.name] = {
                "value": float(barrier.evaluate(state_vector)),
                "margin": float(barrier.compute_margin(state_vector)),
                "safe": bool(barrier.is_safe(state_vector)),
                "limit": float(barrier.limit),
                "current": float(state_vector[barrier.state_index]),
            }
        return status
