"""
AURORA-X Configuration Loader.

Loads YAML configuration and provides typed access to all subsystem settings.
"""

import os
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional


class AuroraConfig:
    """Centralized configuration manager for AURORA-X."""

    _instance: Optional["AuroraConfig"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._config and config_path is None:
            return
        if config_path is None:
            config_path = os.environ.get(
                "AURORA_CONFIG",
                str(Path(__file__).parent.parent / "config" / "default.yaml"),
            )
        self._load(config_path)
        self._setup_logging()

    def _load(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

    def _setup_logging(self):
        log_config_path = Path(__file__).parent.parent / "config" / "logging.yaml"
        if log_config_path.exists():
            # Ensure log directory exists
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_config_path, "r") as f:
                log_cfg = yaml.safe_load(f)
            # Fix log file path to be absolute
            if "handlers" in log_cfg and "file" in log_cfg["handlers"]:
                log_cfg["handlers"]["file"]["filename"] = str(
                    log_dir / "aurora_x.log"
                )
            logging.config.dictConfig(log_cfg)
        else:
            logging.basicConfig(level=logging.INFO)

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path.

        Example:
            config.get("rl.algorithm")  -> "cppo"
            config.get("ingestion.broker.backend")  -> "memory"
        """
        keys = dotpath.split(".")
        val = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def section(self, name: str) -> Dict[str, Any]:
        """Get an entire config section as a dict."""
        return self._config.get(name, {})

    @property
    def mode(self) -> str:
        return self.get("platform.mode", "development")

    @property
    def is_production(self) -> bool:
        return self.mode == "production"

    @property
    def is_development(self) -> bool:
        return self.mode == "development"

    @property
    def is_staging(self) -> bool:
        return self.mode == "staging"

    @property
    def is_maintenance(self) -> bool:
        return self.mode == "maintenance"

    @property
    def is_emergency(self) -> bool:
        return self.mode == "emergency"

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._config = {}

    def __repr__(self):
        return f"AuroraConfig(mode={self.get('platform.mode')}, version={self.get('platform.version')})"
