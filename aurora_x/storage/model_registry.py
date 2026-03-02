"""
AURORA-X Model Registry.

Local file-based or MLflow-backed model versioning.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("aurora_x.storage.model_registry")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelRegistry:
    """Model version management."""

    def __init__(self, config: Dict[str, Any]):
        registry_cfg = config.get("model_registry", {})
        self.backend = registry_cfg.get("backend", "local")
        self.local_path = Path(registry_cfg.get("local_path", "models/"))
        self.local_path.mkdir(parents=True, exist_ok=True)

        self._manifest_path = self.local_path / "manifest.json"
        self._manifest = self._load_manifest()

        logger.info("ModelRegistry initialized (backend=%s, path=%s)", self.backend, self.local_path)

    def _load_manifest(self) -> Dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"models": {}}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def save_model(self, name: str, model, metadata: Optional[Dict] = None):
        """Save a model with versioning."""
        version = self._manifest.get("models", {}).get(name, {}).get("version", 0) + 1
        filename = f"{name}_v{version}.pt"
        filepath = self.local_path / filename

        if TORCH_AVAILABLE and hasattr(model, "state_dict"):
            torch.save(model.state_dict(), filepath)
        else:
            # Fallback: pickle
            import pickle
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

        entry = {
            "version": version,
            "filename": filename,
            "path": str(filepath),
            "saved_at": time.time(),
            "metadata": metadata or {},
        }

        if "models" not in self._manifest:
            self._manifest["models"] = {}
        self._manifest["models"][name] = entry
        self._save_manifest()

        logger.info("Model saved: %s (v%d)", name, version)
        return entry

    def load_model(self, name: str, model_class=None):
        """Load the latest version of a model."""
        entry = self._manifest.get("models", {}).get(name)
        if not entry:
            logger.warning("Model not found: %s", name)
            return None

        filepath = Path(entry["path"])
        if not filepath.exists():
            logger.error("Model file missing: %s", filepath)
            return None

        if TORCH_AVAILABLE and model_class is not None:
            state_dict = torch.load(filepath, weights_only=True)
            model = model_class()
            model.load_state_dict(state_dict)
            return model
        else:
            import pickle
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def list_models(self) -> Dict[str, Any]:
        return self._manifest.get("models", {})
