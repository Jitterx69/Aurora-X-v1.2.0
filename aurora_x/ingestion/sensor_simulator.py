"""
AURORA-X Multi-Sensor Data Simulator.

Generates realistic multi-modal sensor data with configurable fault injection
for industrial assets (gas turbines, pumps, compressors).
"""

import asyncio
import time
import math
import random
import logging
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger("aurora_x.ingestion.simulator")


class SensorPhysicsModel:
    """Physics-based sensor data generation for industrial equipment."""

    def __init__(self, asset_id: str, asset_type: str, seed: int = 42):
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.rng = np.random.RandomState(seed)
        self.t = 0.0
        self.degradation = 0.0  # 0.0 = new, 1.0 = failed

        # Operating state
        self.temperature = 80.0 + self.rng.randn() * 2
        self.vibration_base = 1.0
        self.pressure = 50.0
        self.flow_rate = 100.0
        self.electrical_current = 15.0
        self.acoustic_level = 40.0

        # Fault state
        self.active_faults: Dict[str, float] = {}

    def inject_fault(self, fault_type: str, severity: float = 0.5):
        """Inject a fault with given severity [0, 1]."""
        self.active_faults[fault_type] = min(max(severity, 0.0), 1.0)
        logger.info("Fault injected: %s (severity=%.2f) on %s",
                     fault_type, severity, self.asset_id)

    def clear_faults(self):
        self.active_faults.clear()

    def step(self, dt: float) -> Dict[str, float]:
        """Advance simulation by dt seconds and return sensor readings."""
        self.t += dt
        self.degradation = min(self.degradation + 0.000001 * dt, 1.0)

        readings = {}
        readings["vibration"] = self._vibration(dt)
        readings["thermal"] = self._thermal(dt)
        readings["electrical"] = self._electrical(dt)
        readings["pressure"] = self._pressure(dt)
        readings["flow"] = self._flow(dt)
        readings["acoustic"] = self._acoustic(dt)
        return readings

    def _vibration(self, dt: float) -> float:
        """Damped oscillator model with bearing harmonics."""
        base_freq = 29.5  # Hz (shaft rotation)
        harmonic_2x = math.sin(2 * math.pi * 2 * base_freq * self.t) * 0.3
        harmonic_3x = math.sin(2 * math.pi * 3 * base_freq * self.t) * 0.1
        base = self.vibration_base * (1 + 0.5 * self.degradation)

        noise = self.rng.randn() * 0.1
        vib = base + harmonic_2x + harmonic_3x + noise

        # Fault effects
        if "bearing_wear" in self.active_faults:
            sev = self.active_faults["bearing_wear"]
            # Bearing defect frequency with modulation
            bpfo = math.sin(2 * math.pi * 7.2 * base_freq * self.t)
            vib += sev * 3.0 * abs(bpfo) + sev * self.rng.randn() * 0.5

        if "misalignment" in self.active_faults:
            sev = self.active_faults["misalignment"]
            vib += sev * 2.0 * abs(math.sin(2 * math.pi * base_freq * self.t))

        return max(0.0, vib)

    def _thermal(self, dt: float) -> float:
        """Heat equation approximation with thermal inertia."""
        ambient = 25.0
        heat_gen = 60.0 * (1 + 0.3 * self.degradation)
        thermal_constant = 0.005

        self.temperature += thermal_constant * (
            (ambient + heat_gen) - self.temperature
        ) * dt + self.rng.randn() * 0.05

        if "overheating" in self.active_faults:
            sev = self.active_faults["overheating"]
            self.temperature += sev * 0.5 * dt

        return self.temperature

    def _electrical(self, dt: float) -> float:
        """Motor current model."""
        base = self.electrical_current * (1 + 0.1 * self.degradation)
        ripple = 0.5 * math.sin(2 * math.pi * 50 * self.t)  # 50Hz mains
        noise = self.rng.randn() * 0.2

        current = base + ripple + noise

        if "bearing_wear" in self.active_faults:
            current += self.active_faults["bearing_wear"] * 2.0

        return max(0.0, current)

    def _pressure(self, dt: float) -> float:
        """Pressure dynamics with cavitation effects."""
        base = self.pressure * (1 - 0.05 * self.degradation)
        oscillation = 1.5 * math.sin(2 * math.pi * 0.1 * self.t)
        noise = self.rng.randn() * 0.3

        p = base + oscillation + noise

        if "cavitation" in self.active_faults:
            sev = self.active_faults["cavitation"]
            # Rapid pressure drops
            p -= sev * 10.0 * abs(math.sin(2 * math.pi * 15 * self.t))
            p += sev * self.rng.randn() * 2.0

        return max(0.0, p)

    def _flow(self, dt: float) -> float:
        """Volumetric flow model coupled to pressure."""
        base = self.flow_rate * (1 - 0.08 * self.degradation)
        noise = self.rng.randn() * 0.5
        flow = base + noise

        if "cavitation" in self.active_faults:
            flow *= (1 - 0.3 * self.active_faults["cavitation"])

        return max(0.0, flow)

    def _acoustic(self, dt: float) -> float:
        """Acoustic emission model (dB)."""
        base = self.acoustic_level + 5 * self.degradation
        noise = self.rng.randn() * 1.0
        acoustic = base + noise

        for fault, sev in self.active_faults.items():
            acoustic += sev * 8.0  # All faults increase noise

        return max(0.0, acoustic)


class SensorSimulator:
    """Orchestrates multi-asset sensor simulation.

    Provides a config-only constructor and a synchronous generate() method.
    The orchestrator (main.py) handles the simulation loop.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, SensorPhysicsModel] = {}
        self._running = False

        # Initialize physics models for each asset
        assets = config.get("assets", [{"id": "pump-001", "type": "pump"}])
        for i, asset_cfg in enumerate(assets):
            aid = asset_cfg if isinstance(asset_cfg, str) else asset_cfg.get("id", f"asset-{i}")
            atype = "generic" if isinstance(asset_cfg, str) else asset_cfg.get("type", "generic")
            self.models[aid] = SensorPhysicsModel(
                asset_id=aid,
                asset_type=atype,
                seed=42 + i,
            )
        logger.info("Sensor simulator initialized with %d assets", len(self.models))

    def generate(self, asset_id: str) -> Dict[str, Any]:
        """Generate a single sensor event for the given asset.

        Returns a complete event dict ready for the edge gateway.
        """
        if asset_id not in self.models:
            self.models[asset_id] = SensorPhysicsModel(asset_id, "generic", seed=hash(asset_id) % 10000)

        model = self.models[asset_id]
        dt = 1.0 / self.config.get("sample_rate_hz", 100)

        # Random fault injection
        fault_cfg = self.config.get("fault_injection", {})
        if fault_cfg.get("enabled", False) and random.random() < fault_cfg.get("probability", 0.02) * dt:
            fault_types = fault_cfg.get("types", ["bearing_wear", "misalignment", "cavitation", "overheating"])
            fault = random.choice(fault_types)
            severity = random.uniform(0.2, 0.8)
            model.inject_fault(fault, severity)

        readings = model.step(dt)

        return {
            "asset_id": asset_id,
            "timestamp": time.time(),
            "sample_rate_hz": self.config.get("sample_rate_hz", 100),
            "sensors": readings,
            "metadata": {
                "degradation": model.degradation,
                "active_faults": dict(model.active_faults),
            },
        }

    def inject_fault(self, asset_id: str, fault_type: str, severity: float = 0.5):
        """Inject a fault into a specific asset."""
        if asset_id in self.models:
            self.models[asset_id].inject_fault(fault_type, severity)

