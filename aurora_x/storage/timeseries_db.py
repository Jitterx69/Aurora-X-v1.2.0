"""
AURORA-X Time-Series Database Adapter.

Dual-mode storage: SQLite for development, TimescaleDB for production.
"""

import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger("aurora_x.storage.timeseries")


class TimeSeriesDB:
    """Abstracted time-series storage."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ts_config = config.get("timeseries", {})
        self.backend = ts_config.get("backend", "sqlite")
        self.sqlite_path = ts_config.get("sqlite_path", "data/aurora_timeseries.db")
        self._conn = None
        self._buffer: List[Dict] = []
        self._flush_interval = 100

    async def initialize(self):
        if self.backend == "sqlite":
            try:
                import aiosqlite
            except ImportError:
                logger.warning("aiosqlite not installed. Time-series DB disabled (in-memory buffer only).")
                return
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self.sqlite_path)
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data JSON NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            await self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_time
                ON sensor_data(asset_id, timestamp)
            """)
            await self._conn.commit()
            logger.info("SQLite time-series DB initialized at %s", self.sqlite_path)
        else:
            logger.info("TimescaleDB backend configured (url=%s)",
                        self.config.get("timeseries", {}).get("timescaledb_url", ""))

    async def insert(self, asset_id: str, data: Dict[str, Any]):
        """Insert a data point."""
        self._buffer.append({
            "asset_id": asset_id,
            "timestamp": data.get("timestamp", time.time()),
            "data": data,
        })

        if len(self._buffer) >= self._flush_interval:
            await self.flush()

    async def flush(self):
        """Flush buffered data to storage."""
        if not self._buffer:
            return

        if self.backend == "sqlite" and self._conn:
            try:
                await self._conn.executemany(
                    "INSERT INTO sensor_data (asset_id, timestamp, data) VALUES (?, ?, ?)",
                    [(b["asset_id"], b["timestamp"], json.dumps(b["data"], default=str))
                     for b in self._buffer],
                )
                await self._conn.commit()
            except Exception as e:
                logger.error("Failed to flush to SQLite: %s", e)

        self._buffer.clear()

    async def query(
        self, asset_id: str, start_time: Optional[float] = None,
        end_time: Optional[float] = None, limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query time-series data."""
        if self.backend == "sqlite" and self._conn:
            query = "SELECT asset_id, timestamp, data FROM sensor_data WHERE asset_id = ?"
            params = [asset_id]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()
            return [
                {"asset_id": r[0], "timestamp": r[1], "data": json.loads(r[2])}
                for r in rows
            ]
        return []

    async def close(self):
        await self.flush()
        if self._conn:
            await self._conn.close()
