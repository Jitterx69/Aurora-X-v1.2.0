"""
AURORA-X Fault Detection Engine.

Aggregates residual analysis, ML classifiers, and temporal deep learning
models into a unified probabilistic fault diagnosis system.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List

from aurora_x.fault_detection.residual_analyzer import ResidualAnalyzer
from aurora_x.fault_detection.ml_classifiers import FaultClassifierEnsemble
from aurora_x.fault_detection.temporal_models import TemporalFaultDetector

logger = logging.getLogger("aurora_x.fault_detection.engine")

FAULT_TYPES = ["bearing_wear", "misalignment", "cavitation", "overheating"]
SEVERITY_LEVELS = ["none", "minor", "moderate", "severe", "critical"]


class FaultEngine:
    """Unified fault detection engine combining multiple detection methods.

    Outputs probabilistic fault distributions, severity indices,
    and confidence intervals rather than binary alerts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Detection layers
        self.residual_analyzer = ResidualAnalyzer(config)
        self.ml_classifiers = FaultClassifierEnsemble(config)
        self.temporal_detector = TemporalFaultDetector(config)

        # Ensemble weights
        weights = config.get("ensemble_weights", {})
        self.w_residual = weights.get("residual", 0.2)
        self.w_ml = weights.get("ml", 0.3)
        self.w_temporal = weights.get("temporal", 0.5)

        # Fault history per asset
        self._fault_history: Dict[str, List[Dict]] = {}

        logger.info("FaultEngine initialized (weights: residual=%.1f, ml=%.1f, temporal=%.1f)",
                     self.w_residual, self.w_ml, self.w_temporal)

    def diagnose(
        self,
        asset_id: str,
        state: Dict[str, Any],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run full fault diagnosis pipeline.

        Returns:
            Comprehensive fault report with probabilistic distributions.
        """
        feat_dict = features.get("features", features)

        # --- Layer 1: Residual Analysis ---
        residual_report = self.residual_analyzer.analyze(asset_id, state, features)

        # --- Layer 2: ML Classifiers ---
        feature_vector = self._build_feature_vector(feat_dict)
        ml_report = self.ml_classifiers.predict(feature_vector)

        # --- Layer 3: Temporal Models ---
        temporal_report = self.temporal_detector.predict(asset_id, feature_vector)

        # --- Ensemble Fusion ---
        fault_distribution = self._fuse_predictions(
            residual_report, ml_report, temporal_report
        )

        # --- Severity Assessment ---
        severity = self._assess_severity(fault_distribution, residual_report)

        # --- Confidence Intervals ---
        confidence = self._compute_confidence(
            residual_report, ml_report, temporal_report
        )

        # --- Sensor Attribution (XAI) ---
        attribution = self._attribute_fault(residual_report, fault_distribution)

        # --- Recommendations ---
        recommendations = self._generate_recommendations(
            max(fault_distribution, key=fault_distribution.get),
            severity["index"]
        )

        # Build report
        report = {
            "asset_id": asset_id,
            "timestamp": state.get("timestamp", 0),
            "fault_distribution": fault_distribution,
            "primary_fault": max(fault_distribution, key=fault_distribution.get),
            "primary_probability": max(fault_distribution.values()),
            "severity_index": severity["index"],
            "severity_level": severity["level"],
            "confidence": confidence,
            "attribution": attribution,
            "recommendations": recommendations,
            "residual_analysis": {
                "n_alerts": residual_report["n_alerts"],
                "max_severity": residual_report["max_severity"],
            },
            "ml_prediction": ml_report.get("predicted_fault", "unknown"),
            "temporal_prediction": temporal_report.get("predicted_fault", "unknown"),
            "is_healthy": fault_distribution.get("normal", 0) > 0.7,
            "requires_attention": severity["index"] > 0.3,
            "requires_immediate_action": severity["index"] > 0.7,
        }

        # Track history
        if asset_id not in self._fault_history:
            self._fault_history[asset_id] = []
        self._fault_history[asset_id].append(report)
        if len(self._fault_history[asset_id]) > 1000:
            self._fault_history[asset_id] = self._fault_history[asset_id][-500:]

        return report

    def _attribute_fault(self, residual_report: Dict, fault_dist: Dict) -> Dict[str, float]:
        """Determine which sensors are contributing most to the diagnosis."""
        attribution = {}
        alerts = residual_report.get("alerts", [])
        
        # Calculate total severity to normalize
        total_sev = sum(a["severity"] for a in alerts) or 1.0
        
        for alert in alerts:
            sensor = alert["signal"]
            # Weight attribution by signal severity and fault probability
            attribution[sensor] = (alert["severity"] / total_sev)
            
        return attribution

    def _generate_recommendations(self, fault: str, severity: float) -> List[str]:
        """Generate actionable advice based on detected fault and severity."""
        if fault == "normal" or severity < 0.2:
            return ["No immediate action required.", "Continue routine monitoring."]
            
        recs = {
            "bearing_wear": [
                "Inspect bearing lubrication and temperature.",
                "Schedule vibration analysis for localized defects.",
                "Verify load distribution across shaft bearings."
            ],
            "misalignment": [
                "Perform laser alignment check on coupling.",
                "Check for loose mounting bolts (Soft Foot).",
                "Inspect for thermal expansion stress on piping."
            ],
            "cavitation": [
                "Verify suction pressure and NPSH margins.",
                "Check for blockages in inlet filters/strainers.",
                "Reduce flow rate temporarily to mitigate damage."
            ],
            "overheating": [
                "Check coolant flow and heat exchanger efficiency.",
                "Inspect for internal friction or blocked vents.",
                "Reduce operating load to stabilize temperature."
            ]
        }
        
        base_recs = recs.get(fault, ["Standard diagnostic check required."])
        
        if severity > 0.7:
            return ["CRITICAL: Immediate shutdown recommended.", "Deploy emergency maintenance team."] + base_recs
        elif severity > 0.4:
            return ["WARNING: Schedule maintenance within 24 hours."] + base_recs
        return base_recs

    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to fixed-size numpy vector."""
        # Use a consistent set of features
        keys = sorted([k for k, v in features.items() if isinstance(v, (int, float))])
        vec = np.zeros(50)
        for i, k in enumerate(keys[:50]):
            vec[i] = float(features[k])
        return vec

    def _fuse_predictions(
        self,
        residual: Dict[str, Any],
        ml: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Dict[str, float]:
        """Fuse predictions from all layers using weighted averaging."""
        # Start with base distribution
        faults = {"normal": 0.0, "bearing_wear": 0.0, "misalignment": 0.0,
                  "cavitation": 0.0, "overheating": 0.0}

        # Residual contribution
        residual_severity = residual.get("max_severity", 0.0)
        if residual_severity > 0.1:
            # Distribute severity across alert signals
            for alert in residual.get("alerts", []):
                signal = alert["signal"]
                sev = alert["severity"]
                # Map signals to likely faults
                if signal in ("vibration",):
                    faults["bearing_wear"] += sev * 0.5
                    faults["misalignment"] += sev * 0.3
                elif signal in ("thermal", "temperature"):
                    faults["overheating"] += sev * 0.7
                elif signal in ("pressure", "flow"):
                    faults["cavitation"] += sev * 0.6
                elif signal in ("electrical",):
                    faults["bearing_wear"] += sev * 0.3
                    faults["overheating"] += sev * 0.3
                elif signal in ("acoustic",):
                    faults["bearing_wear"] += sev * 0.3
                    faults["cavitation"] += sev * 0.3
            faults["normal"] = max(0, 1.0 - residual_severity)

            # Normalize residual contribution
            total = sum(faults.values()) or 1.0
            residual_dist = {k: v / total for k, v in faults.items()}
        else:
            residual_dist = {"normal": 0.95, "bearing_wear": 0.0125,
                            "misalignment": 0.0125, "cavitation": 0.0125,
                            "overheating": 0.0125}

        # ML contribution
        ml_dist = ml.get("probabilities", {"normal": 1.0})

        # Temporal contribution
        temporal_dist = temporal.get("probabilities", {"normal": 1.0})

        # Weighted fusion
        result = {}
        all_keys = set(list(residual_dist.keys()) + list(ml_dist.keys()) + list(temporal_dist.keys()))
        for key in all_keys:
            r = residual_dist.get(key, 0.0)
            m = ml_dist.get(key, 0.0)
            t = temporal_dist.get(key, 0.0)
            result[key] = self.w_residual * r + self.w_ml * m + self.w_temporal * t

        # Normalize
        total = sum(result.values()) or 1.0
        result = {k: v / total for k, v in result.items()}

        return result

    def _assess_severity(
        self, fault_distribution: Dict[str, float], residual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute severity index [0, 1] and severity level."""
        fault_prob = 1.0 - fault_distribution.get("normal", 1.0)
        residual_max = residual.get("max_severity", 0.0)

        # Combine fault probability and residual severity
        severity_index = 0.6 * fault_prob + 0.4 * residual_max
        severity_index = np.clip(severity_index, 0.0, 1.0)

        # Map to levels
        if severity_index < 0.1:
            level = "none"
        elif severity_index < 0.3:
            level = "minor"
        elif severity_index < 0.5:
            level = "moderate"
        elif severity_index < 0.7:
            level = "severe"
        else:
            level = "critical"

        return {"index": float(severity_index), "level": level}

    def _compute_confidence(
        self,
        residual: Dict[str, Any],
        ml: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute confidence metrics for the diagnosis."""
        ml_conf = ml.get("confidence", 0.0)
        temporal_conf = temporal.get("confidence", 0.0)

        # Agreement measure
        ml_fault = ml.get("predicted_fault", "normal")
        temporal_fault = temporal.get("predicted_fault", "normal")
        agreement = 1.0 if ml_fault == temporal_fault else 0.5

        overall = (ml_conf + temporal_conf) / 2 * agreement

        return {
            "overall": float(overall),
            "ml_confidence": float(ml_conf),
            "temporal_confidence": float(temporal_conf),
            "agreement": float(agreement),
        }

    def get_fault_history(self, asset_id: str, n: int = 50) -> List[Dict]:
        """Get recent fault reports for an asset."""
        history = self._fault_history.get(asset_id, [])
        return history[-n:]
