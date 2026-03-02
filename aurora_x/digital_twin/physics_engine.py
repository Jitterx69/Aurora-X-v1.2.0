"""
AURORA-X Physics Engine.

4th-order Runge-Kutta ODE solver for rotating machinery dynamics.
Models bearing degradation, thermal accumulation, and vibration resonance.

When aurora_core (Rust/PyO3) is installed, the native implementations
are used for 10-100x speedups. Otherwise, pure-Python fallbacks are used.
"""

import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, Tuple

logger = logging.getLogger("aurora_x.digital_twin.physics")

# ── Try Rust acceleration ──
try:
    from aurora_core import PhysicsEngine as _RustPhysicsEngine
    from aurora_core import RotatingMachineryDynamics as _RustDynamics
    _HAS_RUST = True
    logger.info("Using Rust-accelerated PhysicsEngine (aurora_core)")
except ImportError:
    _HAS_RUST = False
    logger.info("Rust aurora_core not available, using pure-Python PhysicsEngine")


class RungeKutta4:
    """4th-order Runge-Kutta ODE integrator."""

    def __init__(self, dynamics_fn: Callable, dt: float = 0.01):
        self.dynamics = dynamics_fn
        self.dt = dt

    def step(self, state: np.ndarray, t: float, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance state by one time step using RK4.

        Args:
            state: Current state vector.
            t: Current time.
            u: Control input vector (optional).

        Returns:
            New state vector after dt.
        """
        dt = self.dt
        k1 = self.dynamics(state, t, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, t + 0.5 * dt, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, t + 0.5 * dt, u)
        k4 = self.dynamics(state + dt * k3, t + dt, u)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(
        self, state: np.ndarray, t_start: float, t_end: float,
        u: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate over a time range.

        Returns:
            (times, states) arrays.
        """
        n_steps = int((t_end - t_start) / self.dt)
        times = np.linspace(t_start, t_end, n_steps + 1)
        states = np.zeros((n_steps + 1, len(state)))
        states[0] = state

        for i in range(n_steps):
            states[i + 1] = self.step(states[i], times[i], u)

        return times, states


class RotatingMachineryDynamics:
    """Physics model for rotating machinery (turbine/pump/compressor).

    State vector: [theta, omega, T_bearing, T_housing, d_bearing,
                   d_seal, p_inlet, p_outlet]
    Where:
        theta      - shaft angle (rad)
        omega      - shaft angular velocity (rad/s)
        T_bearing  - bearing temperature (°C)
        T_housing  - housing temperature (°C)
        d_bearing  - bearing degradation [0, 1]
        d_seal     - seal degradation [0, 1]
        p_inlet    - inlet pressure (bar)
        p_outlet   - outlet pressure (bar)
    """

    def __init__(self, config: Dict[str, Any]):
        # Physical parameters
        self.J = config.get("inertia", 0.5)          # Moment of inertia (kg·m²)
        self.b = config.get("damping", 0.1)           # Viscous damping
        self.k_thermal = config.get("k_thermal", 0.005)  # Thermal conductivity
        self.T_ambient = config.get("T_ambient", 25.0)    # Ambient temp (°C)
        self.P_rated = config.get("P_rated", 1000.0)       # Rated power (W)

        # Degradation parameters
        self.wear_rate = config.get("wear_rate", 1e-7)
        self.seal_leak_rate = config.get("seal_leak_rate", 5e-8)

    def __call__(
        self, state: np.ndarray, t: float, u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute state derivatives (dynamics function for RK4).

        Args:
            state: [theta, omega, T_bearing, T_housing, d_bearing, d_seal, p_in, p_out]
            t: Current time.
            u: Control input [torque_cmd, cooling_cmd].
        """
        theta, omega, T_b, T_h, d_b, d_s, p_in, p_out = state
        u = u if u is not None else np.zeros(2)

        torque_cmd = u[0] if len(u) > 0 else 0.0
        cooling_cmd = u[1] if len(u) > 1 else 0.0

        # --- Rotational dynamics ---
        # Friction increases with degradation
        friction = self.b * (1 + 3 * d_b) * omega
        d_theta = omega
        d_omega = (torque_cmd - friction) / self.J

        # --- Thermal dynamics ---
        # Heat generation from friction
        Q_friction = friction * abs(omega)
        # Heat generation from power loss
        Q_power = self.P_rated * 0.05 * (1 + d_b * 2)
        # Cooling
        Q_cool = self.k_thermal * (T_b - self.T_ambient) + cooling_cmd * 0.5

        d_T_bearing = (Q_friction + Q_power - Q_cool) / 500.0  # Thermal mass
        d_T_housing = self.k_thermal * (T_b - T_h) - 0.002 * (T_h - self.T_ambient)

        # --- Degradation dynamics ---
        # Bearing wear accelerates with temperature and vibration
        stress_factor = (1 + max(0, T_b - 100) / 50) * (1 + abs(omega) / 100)
        d_bearing = self.wear_rate * stress_factor * (1 + d_b)  # Nonlinear growth

        # Seal degradation (pressure-driven)
        dp = abs(p_in - p_out)
        d_seal = self.seal_leak_rate * dp * (1 + 2 * d_s)

        # --- Pressure dynamics ---
        # Simple pressure model with seal leakage
        d_p_in = -0.001 * (p_in - 50.0)  # Regulated supply
        d_p_out = 0.01 * (omega / 100 * p_in - p_out) - d_s * 0.5  # Output depends on speed

        return np.array([d_theta, d_omega, d_T_bearing, d_T_housing,
                         d_bearing, d_seal, d_p_in, d_p_out])


class PhysicsEngine:
    """High-level physics engine managing multiple asset simulations."""

    def __init__(self, config: Dict[str, Any]):
        self.dt = config.get("dt", 0.01)
        self.solver_type = config.get("solver", "rk4")
        self.config = config

        self._assets: Dict[str, Dict[str, Any]] = {}
        logger.debug("PhysicsEngine initialized (solver=%s, dt=%.4f)", self.solver_type, self.dt)

    def create_asset(self, asset_id: str, params: Optional[Dict] = None) -> np.ndarray:
        """Create a new asset physics simulation."""
        dynamics = RotatingMachineryDynamics(params or {})
        solver = RungeKutta4(dynamics, self.dt)

        # Initial state
        state = np.array([0.0, 157.0, 60.0, 40.0, 0.0, 0.0, 50.0, 30.0])

        self._assets[asset_id] = {
            "dynamics": dynamics,
            "solver": solver,
            "state": state,
            "time": 0.0,
            "history": [],
        }

        logger.debug("Physics model created for asset: %s", asset_id)
        return state

    def step(
        self, asset_id: str, u: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Advance an asset simulation by one time step."""
        if asset_id not in self._assets:
            self.create_asset(asset_id)

        asset = self._assets[asset_id]
        solver = asset["solver"]

        new_state = solver.step(asset["state"], asset["time"], u)

        # Clip degradation to [0, 1]
        new_state[4] = np.clip(new_state[4], 0, 1)  # bearing
        new_state[5] = np.clip(new_state[5], 0, 1)  # seal

        asset["state"] = new_state
        asset["time"] += self.dt
        asset["history"].append(new_state.copy())

        return {
            "shaft_angle": float(new_state[0]),
            "shaft_speed": float(new_state[1]),
            "bearing_temp": float(new_state[2]),
            "housing_temp": float(new_state[3]),
            "bearing_degradation": float(new_state[4]),
            "seal_degradation": float(new_state[5]),
            "inlet_pressure": float(new_state[6]),
            "outlet_pressure": float(new_state[7]),
            "time": asset["time"],
        }

    def simulate(
        self, asset_id: str, duration: float, u: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Simulate an asset for a given duration (deterministic mode)."""
        if asset_id not in self._assets:
            self.create_asset(asset_id)

        asset = self._assets[asset_id]
        solver = asset["solver"]

        times, states = solver.integrate(
            asset["state"], asset["time"], asset["time"] + duration, u
        )

        asset["state"] = states[-1]
        asset["time"] += duration

        return {
            "times": times,
            "states": states,
            "labels": ["shaft_angle", "shaft_speed", "bearing_temp", "housing_temp",
                       "bearing_degradation", "seal_degradation", "inlet_pressure", "outlet_pressure"],
        }

    def get_state(self, asset_id: str) -> Optional[np.ndarray]:
        if asset_id in self._assets:
            return self._assets[asset_id]["state"].copy()
        return None


def create_physics_engine(config: Dict[str, Any]) -> 'PhysicsEngine':
    """Factory: returns Rust-accelerated or Python PhysicsEngine.

    Prefers Rust when aurora_core is available.
    """
    if _HAS_RUST:
        dt = config.get("dt", 0.01)
        logger.info("Creating Rust PhysicsEngine (dt=%.4f)", dt)
        return _RustPhysicsEngine(dt=dt)

    logger.info("Creating Python PhysicsEngine")
    return PhysicsEngine(config)

