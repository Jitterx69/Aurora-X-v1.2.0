"""
AURORA-X Immutable Event Log.

Append-only event log for deterministic replay, forensic audit,
and RL retraining data. Supports both in-memory and file-backed modes.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from collections import deque

logger = logging.getLogger("aurora_x.ingestion.event_log")


class EventLog:
    """Append-only immutable event log with replay support."""

    def __init__(self, max_memory_events: int = 100000, persist_path: Optional[str] = None):
        self._events: deque = deque(maxlen=max_memory_events)
        self._sequence_number = 0
        self._persist_path = persist_path

        if persist_path:
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(persist_path, "a")
            logger.info("Event log persisting to %s", persist_path)
        else:
            self._file = None
            logger.info("Event log in memory-only mode (max=%d)", max_memory_events)

    def append(self, event: Dict[str, Any]) -> int:
        """Append an event and return its sequence number."""
        self._sequence_number += 1

        log_entry = {
            "seq": self._sequence_number,
            "log_timestamp": time.time(),
            "event": event,
        }

        self._events.append(log_entry)

        # Persist to file if configured
        if self._file:
            self._file.write(json.dumps(log_entry, default=str) + "\n")
            if self._sequence_number % 1000 == 0:
                self._file.flush()

        return self._sequence_number

    def replay(
        self,
        start_seq: int = 0,
        end_seq: Optional[int] = None,
        asset_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Replay events from the log for training or audit.

        Args:
            start_seq: Starting sequence number (inclusive).
            end_seq: Ending sequence number (inclusive). None = all.
            asset_id: Filter to specific asset. None = all assets.
        """
        for entry in self._events:
            seq = entry["seq"]
            if seq < start_seq:
                continue
            if end_seq is not None and seq > end_seq:
                break
            if asset_id and entry["event"].get("asset_id") != asset_id:
                continue
            yield entry

    def get_latest(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent events."""
        return list(self._events)[-n:]

    @property
    def size(self) -> int:
        return len(self._events)

    @property
    def latest_sequence(self) -> int:
        return self._sequence_number

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
