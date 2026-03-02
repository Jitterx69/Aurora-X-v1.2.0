"""
AURORA-X ML-Based Fault Classifiers.

Random Forest and SVM classifiers with probabilistic fault outputs.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

logger = logging.getLogger("aurora_x.fault_detection.ml_classifiers")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML classifiers disabled.")


FAULT_CLASSES = ["normal", "bearing_wear", "misalignment", "cavitation", "overheating"]


class FaultClassifierEnsemble:
    """Ensemble of ML classifiers for fault type classification."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._trained = False
        self._scaler = None
        self._rf = None
        self._svm = None
        self._training_buffer: List[Tuple[np.ndarray, int]] = []
        self._min_samples = 100

        if SKLEARN_AVAILABLE:
            rf_cfg = config.get("classifiers", {}).get("random_forest", {})
            svm_cfg = config.get("classifiers", {}).get("svm", {})

            self._rf = RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 100),
                max_depth=rf_cfg.get("max_depth", 10),
                random_state=42,
            )
            self._svm = SVC(
                kernel=svm_cfg.get("kernel", "rbf"),
                C=svm_cfg.get("C", 1.0),
                probability=True,
                random_state=42,
            )
            self._scaler = StandardScaler()

        logger.info("FaultClassifierEnsemble initialized (sklearn=%s)", SKLEARN_AVAILABLE)

    def add_training_sample(self, features: np.ndarray, label: int):
        """Add a labeled sample for online learning."""
        self._training_buffer.append((features, label))

        # Auto-train when we have enough samples
        if len(self._training_buffer) >= self._min_samples and not self._trained:
            self.train()

    def train(self):
        """Train classifiers on accumulated samples."""
        if not SKLEARN_AVAILABLE or len(self._training_buffer) < self._min_samples:
            return

        X = np.array([s[0] for s in self._training_buffer])
        y = np.array([s[1] for s in self._training_buffer])

        # Check we have at least 2 classes
        if len(np.unique(y)) < 2:
            return

        X_scaled = self._scaler.fit_transform(X)

        self._rf.fit(X_scaled, y)
        self._svm.fit(X_scaled, y)

        self._trained = True
        logger.info("Classifiers trained on %d samples", len(self._training_buffer))

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict fault class with probability distribution."""
        if not self._trained or not SKLEARN_AVAILABLE:
            return self._default_prediction()

        X = self._scaler.transform(features.reshape(1, -1))

        # Random Forest prediction
        rf_probs = self._rf.predict_proba(X)[0]
        rf_class = self._rf.predict(X)[0]

        # SVM prediction
        svm_probs = self._svm.predict_proba(X)[0]
        svm_class = self._svm.predict(X)[0]

        # Ensemble averaging
        classes = self._rf.classes_
        ensemble_probs = 0.5 * rf_probs + 0.5 * svm_probs
        ensemble_class = classes[np.argmax(ensemble_probs)]

        # Map class index to fault name
        prob_dict = {}
        for i, cls in enumerate(classes):
            name = FAULT_CLASSES[cls] if cls < len(FAULT_CLASSES) else f"class_{cls}"
            prob_dict[name] = float(ensemble_probs[i])

        return {
            "predicted_class": int(ensemble_class),
            "predicted_fault": FAULT_CLASSES[ensemble_class] if ensemble_class < len(FAULT_CLASSES) else "unknown",
            "probabilities": prob_dict,
            "rf_class": int(rf_class),
            "svm_class": int(svm_class),
            "confidence": float(np.max(ensemble_probs)),
        }

    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when classifiers are not trained."""
        return {
            "predicted_class": 0,
            "predicted_fault": "normal",
            "probabilities": {"normal": 0.9, "bearing_wear": 0.025,
                             "misalignment": 0.025, "cavitation": 0.025,
                             "overheating": 0.025},
            "confidence": 0.0,
            "note": "classifiers_not_trained",
        }

    @property
    def is_trained(self) -> bool:
        return self._trained
