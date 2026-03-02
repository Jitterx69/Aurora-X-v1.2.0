"""
AURORA-X Shell Executor.

Provides a bridge between the API and the system shell for a real terminal experience.
"""

import subprocess
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("aurora_x.api.terminal")

class ShellExecutor:
    """Executes system commands and captures output."""

    def __init__(self, cwd: str = None):
        # Default to project root
        self.cwd = cwd or os.getcwd()
        logger.info("ShellExecutor initialized in %s", self.cwd)

    async def execute(self, command: str) -> Dict[str, Any]:
        """Execute a shell command and return stdout, stderr, and exit code."""
        logger.info("Executing shell command: %s", command)
        
        try:
            # Special case for 'cd' - we simulate it by updating self.cwd
            if command.startswith("cd "):
                new_path = command[3:].strip()
                # Resolve relative to current self.cwd
                full_path = os.path.abspath(os.path.join(self.cwd, new_path))
                if os.path.isdir(full_path):
                    self.cwd = full_path
                    return {
                        "stdout": f"Changed directory to {self.cwd}",
                        "stderr": "",
                        "exit_code": 0,
                        "cwd": self.cwd
                    }
                else:
                    return {
                        "stdout": "",
                        "stderr": f"cd: {new_path}: No such directory",
                        "exit_code": 1,
                        "cwd": self.cwd
                    }

            # Run command via asyncio subprocess
            import asyncio
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                return {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "exit_code": process.returncode,
                    "cwd": self.cwd
                }
            except asyncio.TimeoutExpired:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": "Error: Command timed out after 60 seconds.",
                    "exit_code": 124,
                    "cwd": self.cwd
                }

        except Exception as e:
            logger.error("Shell execution failed: %s", e)
            return {
                "stdout": "",
                "stderr": f"Internal Error: {str(e)}",
                "exit_code": 1,
                "cwd": self.cwd
            }
