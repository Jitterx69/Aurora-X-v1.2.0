"""
AURORA-X Cache Store.

Dual-mode caching: in-memory dict for dev, Redis for production.
"""

import time
import json
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger("aurora_x.storage.cache")


class CacheStore:
    """Key-value cache with TTL support."""

    def __init__(self, config: Dict[str, Any]):
        cache_cfg = config.get("cache", {})
        self.backend = cache_cfg.get("backend", "memory")
        self.ttl = cache_cfg.get("ttl_seconds", 300)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._redis = None

        if self.backend == "redis":
            try:
                import redis
                url = cache_cfg.get("redis_url", "redis://localhost:6379/0")
                self._redis = redis.from_url(url, decode_responses=True)
                logger.info("Redis cache connected at %s", url)
            except Exception as e:
                logger.warning("Redis unavailable (%s), falling back to memory", e)
                self.backend = "memory"

        logger.info("CacheStore initialized (backend=%s, ttl=%ds)", self.backend, self.ttl)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value with optional TTL override."""
        ttl = ttl or self.ttl

        if self.backend == "redis" and self._redis:
            self._redis.setex(key, ttl, json.dumps(value, default=str))
        else:
            self._store[key] = {
                "value": value,
                "expires": time.time() + ttl,
            }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        if self.backend == "redis" and self._redis:
            val = self._redis.get(key)
            return json.loads(val) if val else default
        else:
            entry = self._store.get(key)
            if entry is None:
                return default
            if time.time() > entry["expires"]:
                del self._store[key]
                return default
            return entry["value"]

    def delete(self, key: str):
        if self.backend == "redis" and self._redis:
            self._redis.delete(key)
        else:
            self._store.pop(key, None)

    def keys(self, pattern: str = "*") -> list:
        if self.backend == "redis" and self._redis:
            return self._redis.keys(pattern)
        else:
            if pattern == "*":
                return list(self._store.keys())
            import fnmatch
            return [k for k in self._store if fnmatch.fnmatch(k, pattern)]

    def get_all_latest(self) -> Dict[str, Any]:
        """Get all 'latest:*' entries."""
        result = {}
        for key in self.keys("latest:*"):
            val = self.get(key)
            if val is not None:
                asset_id = key.replace("latest:", "")
                result[asset_id] = val
        return result

    def clear(self):
        """Purge all entries from the cache."""
        if self.backend == "redis" and self._redis:
            self._redis.flushdb()
        else:
            self._store.clear()
        logger.info("CacheStore: all entries purged.")
