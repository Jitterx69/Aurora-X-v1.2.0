"""
AURORA-X Constrained MDP Environment.

Gymnasium-compatible environment wrapping the digital twin for
reinforcement learning-based industrial control optimization.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("aurora_x.rl.environment")

# Action space dimensions
# [maintenance_intensity, speed_adjustment, cooling_adjustment, load_balance]
ACTION_DIM = 4
ACTION_LOW = np.array([0.0, -0.2, 0.0, -0.1])
ACTION_HIGH = np.array([1.0, 0.2, 1.0, 0.1])

# Observation space: state(8) + fault_probs(5) + rul(3) + context(4) = 20
OBS_DIM = 20


class AuroraEnvironment:
    """Constrained Markov Decision Process environment for AURORA-X.

    Observation space:
        - State vector from fusion engine (8 dims)
        - Fault probabilities (5 dims)
        - RUL estimates (mean, std, survival) (3 dims)
        - Context (operating time, energy price, demand, constraint_margin) (4 dims)

    Action space (continuous):
        - maintenance_intensity [0, 1]: How aggressive maintenance is
        - speed_adjustment [-0.2, 0.2]: Shaft speed delta (fraction)
        - cooling_adjustment [0, 1]: Cooling system intensity
        - load_balance [-0.1, 0.1]: Load redistribution factor

    Constraints:
        - Temperature must stay below max_temperature
        - Vibration must stay below max_vibration
        - Pressure must stay within bounds
    """

    def __init__(self, twin_manager, fault_engine, config: Dict[str, Any]):
        self.twin_manager = twin_manager
        self.fault_engine = fault_engine
        self.config = config

        self.action_dim = ACTION_DIM
        self.obs_dim = OBS_DIM
        self.action_low = ACTION_LOW
        self.action_high = ACTION_HIGH

        # Context variables
        self.energy_price = 0.1   # $/kWh (varies)
        self.demand_factor = 1.0  # Production demand multiplier
        self._step_count = 0

        logger.info("AuroraEnvironment initialized (obs_dim=%d, action_dim=%d)",
                     self.obs_dim, self.action_dim)

    def build_observation(
        self, state: Dict[str, Any], fault_report: Dict[str, Any]
    ) -> np.ndarray:
        """Build observation vector from current state and fault analysis."""
        obs = np.zeros(self.obs_dim)

        # State vector (8 dims)
        state_vec = state.get("state_vector", [0] * 8)
        obs[:8] = np.array(state_vec[:8])

        # Fault probabilities (5 dims)
        fault_dist = fault_report.get("fault_distribution", {})
        fault_probs = [
            fault_dist.get("normal", 0.9),
            fault_dist.get("bearing_wear", 0.025),
            fault_dist.get("misalignment", 0.025),
            fault_dist.get("cavitation", 0.025),
            fault_dist.get("overheating", 0.025),
        ]
        obs[8:13] = np.array(fault_probs)

        # RUL estimates (3 dims)
        obs[13] = state.get("degradation", 0.0)
        obs[14] = state.get("confidence", 0.5)
        obs[15] = state.get("trend", 0.0)

        # Context (4 dims)
        obs[16] = self._step_count / 10000.0  # Normalized operating time
        obs[17] = self.energy_price
        obs[18] = self.demand_factor
        obs[19] = fault_report.get("severity_index", 0.0)

        return obs

    def apply_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Apply a control action and return the effect."""
        action = np.clip(action, self.action_low, self.action_high)

        # Convert RL action to physics control input
        # Physics engine expects [torque_cmd, cooling_cmd]
        torque_adjustment = 50.0 * (1 + action[1])  # Base torque + speed adjustment
        cooling_cmd = action[2] * 10.0  # Scale cooling

        control_input = np.array([torque_adjustment, cooling_cmd])

        self._step_count += 1

        # Vary context over time
        self.energy_price = 0.1 + 0.05 * np.sin(self._step_count / 500.0)
        self.demand_factor = 1.0 + 0.2 * np.sin(self._step_count / 1000.0)

        return {
            "control_input": control_input.tolist(),
            "maintenance_intensity": float(action[0]),
            "speed_adjustment": float(action[1]),
            "cooling_adjustment": float(action[2]),
            "load_balance": float(action[3]),
            "step": self._step_count,
        }

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self._step_count = 0
        self.energy_price = 0.1
        self.demand_factor = 1.0
        self.twin_manager.reset()
        return np.zeros(self.obs_dim)

    def get_constraint_values(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Compute constraint values (positive = violated)."""
        safety_cfg = self.config.get("safety", {})
        max_temp = safety_cfg.get("max_temperature", 450.0)
        max_vib = safety_cfg.get("max_vibration", 15.0)
        max_pressure = safety_cfg.get("max_pressure", 120.0)

        temp = state.get("temperature", 0.0)
        vib = state.get("vibration", 0.0)
        pressure = state.get("pressure", 0.0)

        return {
            "temperature_violation": max(0, temp - max_temp),
            "vibration_violation": max(0, vib - max_vib),
            "pressure_violation": max(0, pressure - max_pressure),
            "total_violation": (
                max(0, temp - max_temp) / max_temp
                + max(0, vib - max_vib) / max_vib
                + max(0, pressure - max_pressure) / max_pressure
            ),
        }
