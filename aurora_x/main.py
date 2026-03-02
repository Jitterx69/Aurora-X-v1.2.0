"""
AURORA-X Main Orchestrator.

Initializes all subsystems and runs the closed-loop
sensor → featurize → estimate → twin → fault → RL → safety → actuate pipeline.
"""

import asyncio
import signal
import time
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional

import uvicorn


from aurora_x.security.access_control import SecurityManager, AccessTier
from aurora_x.security.review_manager import ReviewManager
from aurora_x.config import AuroraConfig

logger = logging.getLogger("aurora_x.main")


class AuroraXPlatform:
    """Top-level orchestrator for the AURORA-X platform."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = AuroraConfig(config_path)
        self.running = False

        # --- Subsystem handles (initialized in setup()) ---
        # Ingestion
        self.simulator = None
        self.gateway = None
        self.broker = None
        self.event_log = None
        self.security = SecurityManager()  # Hierarchical access control
        self.reviews = ReviewManager()     # Collaborative review workflow
        self.go_client = None
        self.secure_gateway = None


        # Pipeline
        self.stream_processor = None

        # Estimation
        self.fusion_engine = None

        # Digital Twin
        self.twin_manager = None

        # Fault Detection
        self.fault_engine = None

        # RL
        self.rl_env = None
        self.rl_trainer = None
        self.safety_controller = None

        # Storage
        self.timeseries_db = None
        self.cache = None
        self.model_registry = None

        # Observability
        self.metrics = None
        self.tracing = None

        # API
        self.app = None

        # Internal
        self._tasks = []

    async def setup(self):
        """Initialize all subsystems in dependency order."""
        logger.info("=" * 60)
        logger.info("  AURORA-X Platform — Initialization")
        logger.info("=" * 60)

        # ---- [1/10] Observability ----
        logger.info("[1/10] Initializing observability...")
        from aurora_x.observability.metrics import MetricsCollector
        from aurora_x.observability.tracing import TracingProvider

        self.metrics = MetricsCollector(self.config.section("observability"))
        self.tracing = TracingProvider(self.config.section("observability"))

        # ---- [2/10] Ingestion ----
        logger.info("[2/10] Initializing data ingestion layer...")
        from aurora_x.ingestion.sensor_simulator import SensorSimulator
        from aurora_x.ingestion.edge_gateway import EdgeGateway
        from aurora_x.ingestion.message_broker import create_broker
        from aurora_x.ingestion.event_log import EventLog

        self.simulator = SensorSimulator(self.config.get("ingestion.sensor_simulator", {}))
        self.gateway = EdgeGateway(self.config.section("gateway"))
        self.broker = create_broker(self.config.section("broker"))
        await self.broker.start()

        # [3] Ingestion Layer
        from aurora_x.ingestion.event_log import EventLog
        log_cfg = self.config.section("event_log")
        self.event_log = EventLog(
            max_memory_events=log_cfg.get("max_memory_events", 100000),
            persist_path=log_cfg.get("persist_path")
        )


        # [2.1] Go IPC Client
        from aurora_x.ingestion.go_client import GoServiceClient
        socket_path = self.config.get("gateway.socket_path", "/tmp/aurora_go.sock")
        self.go_client = GoServiceClient(socket_path)
        if self.go_client.connect():
            logger.info("Go Service IPC established via %s", socket_path)
        else:
            logger.warning("Go Service not available, falling back to Python gateway")


        # ---- [3/10] Feature Engineering ----
        logger.info("[3/10] Initializing feature engineering pipeline...")
        from aurora_x.pipeline.stream_processor import create_stream_processor

        self.stream_processor = create_stream_processor(self.config.section("pipeline"))


        # ---- [4/10] State Estimation ----
        logger.info("[4/10] Initializing probabilistic state estimation...")
        from aurora_x.estimation.fusion_engine import FusionEngine

        self.fusion_engine = FusionEngine(self.config.section("estimation"))

        # ---- [5/10] Digital Twin ----
        logger.info("[5/10] Initializing digital twin core...")
        from aurora_x.digital_twin.twin_manager import TwinManager

        self.twin_manager = TwinManager(self.config.section("digital_twin"))

        # ---- [6/10] Fault Detection ----
        logger.info("[6/10] Initializing fault detection engine...")
        from aurora_x.fault_detection.fault_engine import FaultEngine

        self.fault_engine = FaultEngine(self.config.section("fault_detection"))

        # ---- [7/10] RL Subsystem ----
        logger.info("[7/10] Initializing reinforcement learning subsystem...")
        from aurora_x.rl.environment import AuroraEnvironment
        from aurora_x.rl.safety_controller import SafetyController
        from aurora_x.rl.trainer import RLTrainer

        self.safety_controller = SafetyController(self.config.section("safety"))
        self.rl_env = AuroraEnvironment(
            self.twin_manager, self.fault_engine, self.config.section("rl")
        )
        self.rl_trainer = RLTrainer(
            self.rl_env, self.safety_controller, self.config.section("rl")
        )

        # ---- [8/10] Storage & Registry ----
        logger.info("[8/10] Initializing storage and model registry...")
        
        # [8.1] Secure Gateway
        from aurora_x.pipeline.secure_gateway import SecureModelGateway
        self.secure_gateway = SecureModelGateway(secret_key=self.config.get("platform.secret_key", "aurora_x_default_secure_key"))

        from aurora_x.storage.timeseries_db import TimeSeriesDB
        from aurora_x.storage.cache import CacheStore
        from aurora_x.storage.model_registry import ModelRegistry

        self.timeseries_db = TimeSeriesDB(self.config.section("storage"))
        await self.timeseries_db.initialize()
        self.cache = CacheStore(self.config.section("storage"))
        self.model_registry = ModelRegistry(self.config.section("storage"))

        # ---- [9/10] API Server ----
        logger.info("[9/10] Initializing API server...")
        from aurora_x.api.server import create_app

        self.app = create_app(self)

        # ---- [10/10] Done ----
        logger.info("[10/10] ✓ All subsystems initialized")
        logger.info("=" * 60)

    async def run(self):
        """Start all subsystems and enter the main processing loop."""
        self.running = True

        # Register shutdown signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.ensure_future(self.shutdown())
            )

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._processing_loop()))

        # Start sensor simulator if enabled
        sim_cfg = self.config.get("ingestion.sensor_simulator", {})
        if sim_cfg.get("enabled", True):
            self._tasks.append(asyncio.create_task(self._simulator_loop()))

        # Start API server
        api_cfg = self.config.section("api")
        host = api_cfg.get("host", "0.0.0.0")
        port = api_cfg.get("port", 8000)

        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._tasks.append(asyncio.create_task(server.serve()))

        logger.info("AURORA-X Platform RUNNING on http://%s:%d", host, port)
        logger.info("Dashboard: http://%s:%d/dashboard/", host, port)

        # Initial mode trigger
        await self.transition_mode(self.config.mode)

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def on_config_updated(self, updates: Dict[str, Any]):
        """Callback when configuration is changed via API."""
        if "platform" in updates and "mode" in updates["platform"]:
            new_mode = updates["platform"]["mode"]
            await self.transition_mode(new_mode)

    async def transition_mode(self, mode: str):
        """Update platform state based on operational mode."""
        if not self.security.is_mode_allowed(mode):
            logger.error("TRANSITION DENIED: Tier %s cannot activate %s mode.", self.security.active_tier, mode)
            return

        logger.warning(">>> PLATFORM MODE TRANSITION: %s <<<", mode.upper())
        
        if self.event_log:
            self.event_log.append({
                "timestamp": time.time(),
                "asset_id": "system",
                "type": "MODE_TRANSITION",
                "mode": mode
            })

        # 1. Simulator Control
        if self.simulator:
            should_sim = mode in ["development", "maintenance"]
            self.simulator.enabled = should_sim
            logger.info("Simulator state: %s", "ENABLED" if should_sim else "DISABLED")

        # 2. Emergency Halt Logic
        if mode == "emergency":
            logger.critical("!!! EMERGENCY MODE ACTIVE: HALTING LOCAL HIVE LOOPS !!!")
            self.running = False
            # In a real system, we'd trigger a hardware interrupt or safe-shutdown here
        elif not self.running and mode != "emergency":
            # Resume if we were halted (soft-resume)
            logger.info("Resuming platform loops from emergency halt.")
            self.running = True
            self._tasks.append(asyncio.create_task(self._processing_loop()))
            if self.config.get("ingestion.sensor_simulator", {}).get("enabled", True):
                self._tasks.append(asyncio.create_task(self._simulator_loop()))

        # 3. Model Locking in Production
        if mode == "production":
            logger.info("Production mode: Locking model registry for read-only access.")
            if self.model_registry:
                # Mock attribute to signal restriction
                self.model_registry.readonly = True
        else:
            if self.model_registry:
                self.model_registry.readonly = False

    async def _simulator_loop(self):
        """Generate simulated sensor data and publish to broker."""
        interval = self.config.get("ingestion.sensor_simulator", {}).get("interval_ms", 100) / 1000.0
        asset_ids = list(self.simulator.models.keys())

        while self.running:
            for asset_id in asset_ids:
                try:
                    raw_event = self.simulator.generate(asset_id)
                    
                    # Route through Go service if available
                    validated = None
                    if self.go_client and self.go_client.is_connected:
                        validated = self.go_client.send_event(raw_event)
                    
                    # Fallback to Python gateway
                    if validated is None:
                        validated = self.gateway.process(raw_event)
                        
                    if validated:
                        await self.broker.publish("sensor_data", asset_id, validated)
                except Exception as e:
                    logger.error("Simulator error for %s: %s", asset_id, e)

            await asyncio.sleep(interval)

    async def _processing_loop(self):
        """Main processing pipeline loop using Secure Model Gateway."""
        logger.info("Main processing loop started with Secure Gateway.")
        while self.running:
            try:
                # 1. Batch consume sensor data from broker
                batch = await self.broker.consume_batch("sensor_data", max_batch=20, timeout_ms=100)
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                
                # Simple Anomaly/Unusual Activity Detection
                if len(batch) > 50:
                    logger.warning("UNUSUAL ACTIVITY: High-frequency event burst detected (%d events)", len(batch))
                    self.event_log.append({
                        "timestamp": time.time(),
                        "asset_id": "security_monitor",
                        "type": "UNUSUAL_TRAFFIC",
                        "details": f"Burst of {len(batch)} events"
                    })

                for event in batch:
                    t_start = time.time()
                    asset_id = event.get("asset_id", "unknown")
                    
                    # 2. Log immutably (WAL)
                    self.event_log.append(event)
                    
                    # 3. Securely process through ML/RL pipeline
                    result = await self._secure_process(asset_id, event)
                    if result:
                        # 4. Persistence & Metrics
                        composite = {
                            "timestamp": time.time(),
                            "event": event,
                            **result
                        }
                        self.cache.set(f"latest:{asset_id}", composite, ttl=60)
                        await self.timeseries_db.insert(asset_id, composite)
                        
                        latency = time.time() - t_start
                        self.metrics.record_event_processed(latency)
                        self.metrics.record_fault_severity(
                            asset_id, result.get("fault_report", {}).get("severity_index", 0)
                        )

            except Exception as e:
                logger.error("Processing loop error: %s", e, exc_info=True)
                await asyncio.sleep(0.1)

    async def _secure_process(self, asset_id: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async wrapper for the heavy synchronous secure pipeline."""
        return await asyncio.to_thread(self._secure_process_sync, asset_id, event)

    def _secure_process_sync(self, asset_id: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Synchronous implementation of the secure pipeline.
        Runs in a separate thread to avoid blocking the event loop.
        """
        try:
            # 1. Featurize
            features = self.stream_processor.process(event)

            # 2. State Estimation
            def estimation_task(asset_id, features):
                return self.fusion_engine.update(asset_id, features)

            secure_est = self.secure_gateway.secure_execute(
                estimation_task, {"asset_id": asset_id, "features": features}
            )
            state_report = self.secure_gateway.decrypt_result(secure_est)
            state = state_report.get("fused_state", {})

            # 3. Sync Twin
            self.twin_manager.update(asset_id, state)

            # 4. Fault Detection
            def fault_task(asset_id, state, features):
                return self.fault_engine.diagnose(asset_id, state, features)

            secure_fault = self.secure_gateway.secure_execute(
                fault_task, {"asset_id": asset_id, "state": state, "features": features}
            )
            fault_report = self.secure_gateway.decrypt_result(secure_fault)

            # 5. RL Optimization
            def rl_task(state, fault_report):
                obs = self.rl_env.build_observation(state, fault_report)
                action = self.rl_trainer.select_action(obs, deterministic=True)
                return self.rl_env.apply_action(action)

            secure_rl = self.secure_gateway.secure_execute(
                rl_task, {"state": state, "fault_report": fault_report}
            )
            rl_result = self.secure_gateway.decrypt_result(secure_rl)

            # 6. Safety Validation
            safe_action = self.safety_controller.validate(
                np.array(rl_result["control_input"]), 
                state, 
                self.twin_manager.get_twin(asset_id)
            )

            return {
                "features": features,
                "state": state,
                "fault_report": fault_report,
                "rl_action": rl_result,
                "safe_action": safe_action.tolist(),
                "secure_status": "verified"
            }

        except Exception as e:
            logger.error("Secure process error for %s: %s", asset_id, e)
            return None


    async def reboot(self):
        """Soft reboot: Stop processes, clear state, and restart loops."""
        logger.warning("SOFT REBOOT INITIATED: Re-initializing AURORA-X Subsystems...")
        
        # 1. Stop processing/simulation loops
        self.running = False
        await asyncio.sleep(0.5) # Allow loops to finish current iteration
        
        # 2. Reset state
        if self.cache:
            self.cache.clear()
            logger.info("System cache purged for reboot.")
            
        # 3. Resume loops
        self.running = True
        self._tasks.append(asyncio.create_task(self._processing_loop()))
        
        sim_cfg = self.config.get("ingestion.sensor_simulator", {})
        if sim_cfg.get("enabled", True):
            self._tasks.append(asyncio.create_task(self._simulator_loop()))
            
        logger.info("AURORA-X platform successfully REBOOTED.")
        return True

    async def shutdown(self):
        """Gracefully shut down all subsystems."""
        logger.info("Shutting down AURORA-X platform...")
        self.running = False

        for task in self._tasks:
            task.cancel()

        if self.broker:
            await self.broker.stop()
        if self.timeseries_db:
            await self.timeseries_db.close()

        logger.info("AURORA-X platform stopped.")


async def main():
    platform = AuroraXPlatform()
    await platform.setup()
    await platform.run()


if __name__ == "__main__":
    asyncio.run(main())
