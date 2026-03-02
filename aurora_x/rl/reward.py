"""
AURORA-X Multi-Objective Reward Function.

Balances operational efficiency, degradation penalties,
risk exposure, and energy cost in the RL optimization.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("aurora_x.rl.reward")


class RewardFunction:
    """Multi-component reward function for industrial control optimization.

    R(s, a) = w_eff * R_efficiency
            + w_deg * R_degradation
            + w_risk * R_risk
            + w_energy * R_energy

    All components are designed so that higher is better.
    """

    def __init__(self, config: Dict[str, Any]):
        weights = config.get("reward_weights", {})
        self.w_efficiency = weights.get("efficiency", 1.0)
        self.w_degradation = weights.get("degradation_penalty", -2.0)
        self.w_risk = weights.get("risk_cost", -5.0)
        self.w_energy = weights.get("energy_cost", -0.5)

        logger.info("RewardFunction initialized (eff=%.1f, deg=%.1f, risk=%.1f, energy=%.1f)",
                     self.w_efficiency, self.w_degradation, self.w_risk, self.w_energy)

    def compute(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        fault_report: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute the complete reward and its components.

        Args:
            state: Current state estimate.
            action: Action taken [maintenance, speed, cooling, load].
            next_state: State after action.
            fault_report: Fault detection results.
            context: Operating context (energy price, demand, etc.).

        Returns:
            Dict with total reward and component breakdown.
        """
        # --- R_efficiency: Production output relative to target ---
        flow = next_state.get("flow", 100)
        target_flow = 100.0 * context.get("demand_factor", 1.0)
        r_efficiency = 1.0 - abs(flow - target_flow) / target_flow
        r_efficiency = max(0, r_efficiency)

        # --- R_degradation: Penalty for degradation acceleration ---
        deg_current = state.get("degradation", 0.0)
        deg_next = next_state.get("degradation", 0.0)
        deg_delta = deg_next - deg_current

        # Penalize positive degradation growth
        r_degradation = -max(0, deg_delta) * 100.0

        # Bonus for maintenance that reduces degradation
        maintenance_intensity = float(action[0]) if len(action) > 0 else 0.0
        if maintenance_intensity > 0.5:
            r_degradation += maintenance_intensity * 0.1

        # --- R_risk: Penalty for fault risk exposure ---
        severity = fault_report.get("severity_index", 0.0)
        normal_prob = fault_report.get("fault_distribution", {}).get("normal", 1.0)

        r_risk = -(severity ** 2)  # Quadratic penalty

        # Extra penalty for critical states
        if fault_report.get("requires_immediate_action", False):
            r_risk -= 1.0

        # --- R_energy: Energy cost minimization ---
        speed_adj = float(action[1]) if len(action) > 1 else 0.0
        cooling_adj = float(action[2]) if len(action) > 2 else 0.0
        energy_price = context.get("energy_price", 0.1)

        energy_consumption = (1 + speed_adj) * 0.5 + cooling_adj * 0.3
        r_energy = -energy_consumption * energy_price

        # --- Total Reward ---
        total = (
            self.w_efficiency * r_efficiency
            + self.w_degradation * r_degradation
            + self.w_risk * r_risk
            + self.w_energy * r_energy
        )

        return {
            "total": float(total),
            "efficiency": float(r_efficiency),
            "degradation": float(r_degradation),
            "risk": float(r_risk),
            "energy": float(r_energy),
            "components": {
                "w_eff * R_eff": float(self.w_efficiency * r_efficiency),
                "w_deg * R_deg": float(self.w_degradation * r_degradation),
                "w_risk * R_risk": float(self.w_risk * r_risk),
                "w_energy * R_energy": float(self.w_energy * r_energy),
            },
        }

    def compute_cost(
        self, state: Dict[str, Any], constraints: Dict[str, float]
    ) -> float:
        """Compute constraint cost for CMDP (used by Lagrangian methods).

        Returns positive cost when constraints are violated.
        """
        total_violation = constraints.get("total_violation", 0.0)
        return float(total_violation)
