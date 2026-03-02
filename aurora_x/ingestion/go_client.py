"""
AURORA-X Go Service Client.

Unix domain socket client for high-performance IPC with Go-based ingestion services.
"""

import json
import socket
import logging
import time
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("aurora_x.ingestion.go_client")

class GoServiceClient:
    """Client for communicating with Go services over Unix Domain Sockets."""

    def __init__(self, socket_path: str = "/tmp/aurora_go.sock", timeout: float = 1.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to the Go service."""
        if self._connected:
            return True

        if not os.path.exists(self.socket_path):
            logger.debug("Go service socket not found at %s", self.socket_path)
            return False

        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(self.socket_path)
            self._connected = True
            logger.info("Connected to Go service at %s", self.socket_path)
            return True
        except Exception as e:
            logger.error("Failed to connect to Go service: %s", e)
            self._connected = False
            return False

    def send_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a sensor event to the Go gateway and wait for validation."""
        if not self._connected and not self.connect():
            return None

        try:
            # Send newline-terminated JSON
            data = json.dumps(event) + "\n"
            self.sock.sendall(data.encode())

            # Receive response
            resp_data = self.sock.recv(4096)
            if not resp_data:
                self.close()
                return None

            return json.loads(resp_data.decode())
        except Exception as e:
            logger.error("IPC error with Go service: %s", e)
            self.close()
            return None

    def close(self):
        """Close the socket connection."""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
