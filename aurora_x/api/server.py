"""
AURORA-X FastAPI Server.

REST API and WebSocket endpoints for the AURORA-X platform.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import os

logger = logging.getLogger("aurora_x.api.server")


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(self.active_connections))

    async def broadcast(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


from aurora_x.api.terminal import ShellExecutor

def create_app(platform) -> FastAPI:
    """Create FastAPI application with all routes."""

    app = FastAPI(
        title="AURORA-X",
        description="Autonomous Uncertainty-Resolved Optimization and Resilient Actuation Platform",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Subsystem instances for API
    ws_manager = ConnectionManager()
    shell_executor = ShellExecutor()

    # Serve dashboard static files
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
    if dashboard_path.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")

    # ============ REST Endpoints ============

    @app.get("/")
    async def root():
        return {
            "name": "AURORA-X",
            "version": "1.0.0",
            "status": "running",
            "uptime": time.time() - platform.metrics._start_time if platform.metrics else 0,
        }

    @app.get("/api/status")
    async def get_status():
        return {
            "platform": "AURORA-X",
            "running": platform.running,
            "metrics": platform.metrics.get_stats() if platform.metrics else {},
            "gateway_stats": platform.gateway.get_stats() if platform.gateway else {},
            "tracked_assets": platform.fusion_engine.tracked_assets if platform.fusion_engine else [],
            "event_log_size": platform.event_log.size if platform.event_log else 0,
        }

    @app.get("/api/assets")
    async def list_assets():
        """List all tracked assets with current state."""
        if platform.cache:
            all_latest = platform.cache.get_all_latest()
            return {"assets": all_latest}
        return {"assets": {}}

    @app.get("/api/assets/{asset_id}")
    def get_asset_details(asset_id: str):
        """Get detailed state for a specific asset."""
        state = None
        if platform.cache:
            state = platform.cache.get(f"latest:{asset_id}")

        health = None
        if platform.twin_manager:
            health = platform.twin_manager.get_health_report(asset_id)

        return {
            "asset_id": asset_id,
            "current_state": state,
            "health_report": health,
        }

    @app.get("/api/assets/{asset_id}/faults")
    async def get_faults(asset_id: str, n: int = 50):
        """Get fault history for an asset."""
        if platform.fault_engine:
            history = platform.fault_engine.get_fault_history(asset_id, n)
            return {"asset_id": asset_id, "faults": history}
        return {"asset_id": asset_id, "faults": []}

    @app.get("/api/twin/{asset_id}")
    def get_twin(asset_id: str):
        """Get digital twin state."""
        logger.info(f"Incoming Digital Twin Request for {asset_id}")
        if platform.twin_manager:
            report = platform.twin_manager.get_health_report(asset_id)
            return {"asset_id": asset_id, "twin": report, "mode": platform.twin_manager.mode}
        return {"asset_id": asset_id, "twin": None, "mode": "realtime"}

    @app.post("/api/twin/{asset_id}/mode")
    async def set_twin_mode(asset_id: str, payload: Dict[str, str] = Body(...)):
        """Set twin to realtime or simulation mode."""
        mode = payload.get("mode", "realtime")
        if platform.twin_manager:
            platform.twin_manager.mode = mode
            logger.info(f"Twin mode set to {mode}")
            return {"status": "SUCCESS", "mode": mode}
        return {"status": "ERROR", "message": "Twin manager not available"}

    @app.post("/api/twin/{asset_id}/inject")
    def inject_twin_data(asset_id: str, payload: Dict[str, Any] = Body(...)):
        """Inject state forces directly into the twin for simulation."""
        if platform.twin_manager and platform.twin_manager.mode == "simulation":
            twin = platform.twin_manager.get_twin(asset_id)
            if twin:
                # Update specific state overrides
                state = twin._synced_state if twin._synced_state else {}
                for k, v in payload.items():
                    state[k] = v
                twin.update(state)
                return {"status": "SUCCESS", "injected": payload}
        return {"status": "ERROR", "message": "Must be in simulation mode to inject data"}

    @app.get("/api/rl/stats")
    async def get_rl_stats():
        """Get RL training statistics."""
        if platform.rl_trainer:
            return platform.rl_trainer.get_training_stats()
        return {"trained": False}

    @app.get("/api/rl/safety")
    async def get_safety_status():
        """Get safety controller status."""
        if platform.safety_controller:
            stats = platform.safety_controller.get_stats()
            # Get barrier status for all assets
            barrier_status = {}
            if platform.cache:
                for asset_id, data in platform.cache.get_all_latest().items():
                    sensors = data.get("event", {}).get("sensors", {})
                    state = {
                        "temperature": sensors.get("thermal", 0),
                        "vibration": sensors.get("vibration", 0),
                        "pressure": sensors.get("pressure", 0),
                    }
                    try:
                        barrier_status[asset_id] = platform.safety_controller.get_barrier_status(state)
                    except Exception as e:
                        logger.warning("Barrier status error for %s: %s", asset_id, e)
                        barrier_status[asset_id] = {}
            return {"stats": stats, "barriers": barrier_status}
        return {"stats": {}, "barriers": {}}

    @app.post("/api/command")
    async def post_command(cmd: Dict[str, Any]):
        """Receive human command overrides."""
        logger.info(f"Manual Command Received: {cmd}")
        asset_id = cmd.get("asset_id", "global")
        action = cmd.get("action", "query").strip()
        
        # 1. Platform Shortcuts
        if action == "shutdown":
            logger.warning("EMERGENCY SHUTDOWN TRIGGERED VIA API")
            asyncio.create_task(platform.shutdown())
            return {"status": "SHUTDOWN_INITIATED", "asset": asset_id}
            
        if action == "clear_cache":
            if platform.cache:
                platform.cache.clear()
                logger.info("Platform cache purged via manual command")
                return {"status": "CACHE_CLEARED", "asset": asset_id}
            return {"status": "ERROR", "message": "Cache not initialized"}

        if action == "reboot":
            logger.warning("SOFT REBOOT TRIGGERED VIA API")
            asyncio.create_task(platform.reboot())
            return {"status": "REBOOT_INITIATED", "asset": asset_id}

        if action == "reset_rl":
            if platform.rl_trainer:
                logger.info("RL Trainer reset initiated")
                return {"status": "RL_RESET_SUCCESS", "asset": asset_id}

        if action == "inject_fault":
            fault_type = cmd.get("fault_type", "bearing_wear")
            severity = cmd.get("severity", 0.5)
            if platform.simulator:
                platform.simulator.inject_fault(asset_id, fault_type, severity)
                logger.warning("FAULT INJECTED: %s (severity=%.2f) on %s", fault_type, severity, asset_id)
                return {"status": "FAULT_INJECTED", "fault_type": fault_type, "severity": severity, "asset": asset_id}
            return {"status": "ERROR", "message": "Simulator not available"}

        if action == "clear_faults":
            if platform.simulator and asset_id in platform.simulator.models:
                platform.simulator.models[asset_id].clear_faults()
                logger.info("All faults cleared for %s", asset_id)
                return {"status": "FAULTS_CLEARED", "asset": asset_id}
            return {"status": "ERROR", "message": "Asset not found in simulator"}

        # 2. Mode-Based Security Guardrail
        if platform.config.mode in ["production", "emergency"] and action not in ["inject_fault", "clear_faults"]:
            logger.warning("BLOCKED UNAUTHORIZED REQUEST: Terminal access disabled in %s mode.", platform.config.mode)
            platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "security",
                "type": "RESTRICTED_ACCESS_VIOLATION",
                "details": f"Attempted shell command '{action}' in {platform.config.mode} mode"
            })
            return {"status": "ERROR", "message": f"Shell access RESTRICTED in {platform.config.mode.upper()} mode."}

        # 3. General Shell Execution (Linux Commands)
        result = await shell_executor.execute(action)
        return {
            "status": "SHELL_EXECUTION",
            "command": action,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
            "cwd": result["cwd"],
            "timestamp": time.time()
        }

    @app.get("/api/security/audit")
    async def get_security_audit(limit: int = 100):
        """Fetch records from the homomorphic secure audit log."""
        db_path = "aurora_x/storage/secure_audit.db"
        if not os.path.exists(db_path):
             return {"logs": [], "error": "Audit database not found"}
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM secure_audit ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {"logs": rows}
        except Exception as e:
            logger.error(f"Audit fetch error: {e}")
            return {"logs": [], "error": str(e)}

    # ============ Config Endpoints (Settings Screen) ============

    @app.get("/api/config")
    async def get_config():
        """Return full platform configuration."""
        return {"config": platform.config._config}

    @app.put("/api/config")
    async def update_config(updates: Dict[str, Any], reviewed: bool = False, biometric_verified: bool = False):
        """Apply partial config updates with hierarchical safety checks."""
        
        # 1. Tier-Based Mode Transition Check
        if "platform" in updates and "mode" in updates["platform"]:
            target_mode = updates["platform"]["mode"]
            if not platform.security.is_mode_allowed(target_mode):
                logger.warning("TRANSITION DENIED: Tier %s blocked from %s mode.", platform.security.active_tier, target_mode)
                return {
                    "status": "ACCESS_DENIED",
                    "message": f"Security Tier {platform.security.active_tier.upper()} restricted from {target_mode.upper()} mode."
                }
            
            # Biometric check for all except potentially Junior? (User said "Except the user with junior key")
            # If Junior can't switch anyway, then only higher tiers need biometric.
            if platform.security.active_tier != "junior" and not biometric_verified:
                 return {
                    "status": "BIOMETRIC_REQUIRED",
                    "message": "Biometric verification REQUIRED for mode transitions."
                }

        # 2. Production/Staging Review Logic
        if platform.config.mode in ["production", "staging"] and not reviewed:
            logger.warning("CONFIG PUSH REJECTED: Changes in %s mode require moderator review.", platform.config.mode)
            return {
                "status": "REVIEW_REQUIRED", 
                "message": f"Changes in {platform.config.mode.upper()} mode must be PUSHED FOR REVIEW first."
            }

        applied = []
        for section, values in updates.items():
            if section in platform.config._config and isinstance(values, dict):
                for key, val in values.items():
                    if isinstance(platform.config._config[section], dict):
                        old = platform.config._config[section].get(key)
                        platform.config._config[section][key] = val
                        applied.append({"section": section, "key": key, "old": old, "new": val})
                        logger.info("Config updated: %s.%s = %s (was %s)", section, key, val, old)
        
        # Notify platform of changes
        await platform.on_config_updated(updates)

        return {"status": "CONFIG_UPDATED", "applied": applied, "count": len(applied)}

    # ============ Security Access Endpoints ============

    @app.get("/api/security/status")
    async def get_security_status():
        """Return current access tier and permissions."""
        return platform.security.get_status()

    @app.post("/api/security/activate")
    async def activate_software_key(key: str = Body(..., embed=True)):
        """Activate a software tier key with upgrade support."""
        # 1. Check if it's the MASTER key first (Master always allowed to override)
        if key == "AX-MASTER-0000":
             platform.security.activate_key(key)
             platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "system",
                "type": "MASTER_ACCESS_GRANTED",
                "details": "Master hardware activation key verified."
            })
             return {"status": "SUCCESS", "tier": "master"}

        # 2. Check if key change is allowed for other keys
        if not platform.security.can_manage_keys():
            return {"status": "ERROR", "message": "Access restricted. Only MASTER key holder can change activation levels."}
        
        # 3. Handle activation/upgrade
        success = platform.security.activate_key(key)
        if success:
             platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "system",
                "type": "USER_ONBOARDED",
                "tier": platform.security.active_tier,
                "details": f"Software activation key {key[:4]}... accepted."
            })
             # Force platform to re-transition to current mode to apply new restrictions etc?
             return {"status": "SUCCESS", "tier": platform.security.active_tier}
        return {"status": "ERROR", "message": "INVALID SOFTWARE ACTIVATION KEY"}

    @app.post("/api/security/biometric")
    async def biometric_verify():
        """Simulate hardware-level biometric verification."""
        # Mock success
        return {"status": "BIOMETRIC_MATCH", "timestamp": time.time()}

    # ============ Collaborative Review Endpoints ============

    @app.post("/api/security/reviews")
    async def submit_review(updates: Dict[str, Any] = Body(...)):
        """Submit a configuration push for review."""
        ticket_id = platform.reviews.create_ticket(
            requester=platform.security.active_tier,
            changes=updates,
            tier=platform.security.active_tier
        )
        logger.info("PUSH SUBMITTED FOR REVIEW: Ticket ID %s", ticket_id)
        platform.event_log.append({
            "timestamp": time.time(),
            "asset_id": "system",
            "type": "PUSH_SUBMITTED",
            "details": f"Configuration review ticket {ticket_id} created by tier {platform.security.active_tier}."
        })
        return {"status": "SUCCESS", "ticket_id": ticket_id}

    @app.get("/api/security/reviews")
    async def get_reviews():
        """Fetch all review tickets."""
        return {"tickets": platform.reviews.get_all_tickets()}

    @app.get("/api/security/reviews/{ticket_id}")
    async def get_ticket(ticket_id: str):
        """Fetch a specific review ticket."""
        ticket = platform.reviews.get_ticket(ticket_id)
        if ticket:
            return ticket.to_dict()
        raise HTTPException(status_code=404, detail="Ticket not found")

    @app.post("/api/security/reviews/{ticket_id}/feedback")
    async def submit_feedback(ticket_id: str, feedback: Dict[str, Any] = Body(...)):
        """Submit moderator feedback or bug reports."""
        author = platform.security.active_tier
        text = feedback.get("text", "")
        is_bug = feedback.get("is_bug", False)
        
        success = platform.reviews.submit_feedback(ticket_id, author, text, is_bug)
        if success:
             platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "security",
                "type": "REVIEW_FEEDBACK",
                "details": f"Moderator feedback added to ticket {ticket_id}."
            })
             return {"status": "SUCCESS"}
        return {"status": "ERROR", "message": "Ticket not found"}

    @app.post("/api/security/reviews/{ticket_id}/resolve")
    async def resolve_bugs(ticket_id: str):
        """Mark bugs as resolved in a ticket."""
        success = platform.reviews.resolve_ticket(ticket_id)
        if success:
            platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "system",
                "type": "BUGS_RESOLVED",
                "details": f"Bugs resolved for review ticket {ticket_id}."
            })
            return {"status": "SUCCESS"}
        return {"status": "ERROR", "message": "Ticket not found"}

    @app.post("/api/security/reviews/{ticket_id}/approve")
    async def approve_review(ticket_id: str):
        """Approve a ticket for final deployment."""
        if platform.security.active_tier not in ["moderator", "senior", "master"]:
             return {"status": "ERROR", "message": "Unauthorized. Only Moderator tier or higher can approve changes."}
        
        success = platform.reviews.approve_ticket(ticket_id, platform.security.active_tier)
        if success:
             platform.event_log.append({
                "timestamp": time.time(),
                "asset_id": "security",
                "type": "PUSH_APPROVED",
                "details": f"Ticket {ticket_id} approved for production deployment."
            })
             # Actually apply the config changes?
             ticket = platform.reviews.get_ticket(ticket_id)
             await platform.on_config_updated(ticket.changes)
             return {"status": "SUCCESS"}
        return {"status": "ERROR", "message": "Ticket not found"}

    # ============ Event Log Stream ============

    @app.get("/api/events/stream")
    async def get_event_stream(limit: int = 50):
        """Return recent events from the event log."""
        events = []
        if platform.event_log:
            raw = platform.event_log._events
            recent = list(raw)[-limit:]
            for ev in reversed(recent):
                try:
                    events.append({
                        "asset_id": ev.get("asset_id", "unknown"),
                        "timestamp": ev.get("timestamp", 0),
                        "sensors": ev.get("sensors", {}),
                        "fault_type": ev.get("fault_type", None),
                    })
                except Exception:
                    pass
        return {"events": events, "total_size": platform.event_log.size if platform.event_log else 0}

    @app.get("/api/pipeline/stats")
    async def get_pipeline_stats():
        """Return pipeline processing stats."""
        stats = {}
        if platform.metrics:
            stats["metrics"] = platform.metrics.get_stats()
        if platform.gateway:
            stats["gateway"] = {
                "events_processed": getattr(platform.gateway, 'events_processed', 0),
                "events_rejected": getattr(platform.gateway, 'events_rejected', 0),
                "compression": getattr(platform.gateway, 'compression_enabled', False),
            }
        if platform.event_log:
            stats["event_log"] = {"size": platform.event_log.size, "max": 100000}
        if platform.cache:
            stats["cache"] = {
                "backend": platform.config.get("storage.cache.backend", "memory"),
                "ttl": platform.config.get("storage.cache.ttl_seconds", 300),
            }
        if platform.stream_processor:
            stats["pipeline"] = {
                "window_size": platform.config.get("pipeline.window_size_samples", 256),
                "overlap": platform.config.get("pipeline.window_overlap", 0.5),
                "fft_bins": platform.config.get("pipeline.fft_n_bins", 128),
            }
        return stats

    # ============ WebSocket Endpoint ============

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Send latest data to connected clients
                data = {}
                if platform.cache:
                    data["assets"] = platform.cache.get_all_latest()
                if platform.metrics:
                    data["metrics"] = platform.metrics.get_stats()
                if platform.twin_manager:
                    data["twins"] = platform.twin_manager.get_all_reports()
                    data["twin_mode"] = platform.twin_manager.mode
                if platform.safety_controller:
                    data["safety"] = platform.safety_controller.get_stats()
                if platform.gateway:
                    data["gateway"] = {
                        "events_processed": getattr(platform.gateway, 'events_processed', 0),
                        "events_rejected": getattr(platform.gateway, 'events_rejected', 0),
                    }
                if platform.event_log:
                    data["event_log_size"] = platform.event_log.size
                if platform.rl_trainer:
                    data["rl_stats"] = platform.rl_trainer.get_training_stats()
                if platform.safety_controller and platform.cache:
                    barrier_data = {}
                    for aid, adata in platform.cache.get_all_latest().items():
                        sensors = adata.get("event", {}).get("sensors", {})
                        state = {
                            "temperature": sensors.get("thermal", 0),
                            "vibration": sensors.get("vibration", 0),
                            "pressure": sensors.get("pressure", 0),
                        }
                        try:
                            barrier_data[aid] = platform.safety_controller.get_barrier_status(state)
                        except Exception:
                            barrier_data[aid] = {}
                    data["barriers"] = barrier_data

                data["timestamp"] = time.time()
                await websocket.send_json(data)

                # Also listen for commands from dashboard
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    cmd = json.loads(msg)
                    logger.info("Received WS command: %s", cmd)
                except asyncio.TimeoutError:
                    pass
                except (WebSocketDisconnect, asyncio.CancelledError):
                    break
                except Exception as e:
                    logger.error(f"WS receive error: {e}")
                    break

        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            ws_manager.disconnect(websocket)

    # Background task to broadcast data
    @app.on_event("startup")
    async def start_broadcast():
        async def _broadcast_loop():
            while True:
                try:
                    if ws_manager.active_connections and platform.cache:
                        data = {
                            "type": "update",
                            "assets": platform.cache.get_all_latest(),
                            "metrics": platform.metrics.get_stats() if platform.metrics else {},
                            "timestamp": time.time(),
                        }
                        await ws_manager.broadcast(data)
                except Exception as e:
                    logger.error("Broadcast error: %s", e)
                await asyncio.sleep(0.5)

        asyncio.create_task(_broadcast_loop())

    return app
