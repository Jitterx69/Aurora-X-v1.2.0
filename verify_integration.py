import logging
import sys
import os
import json
import socket
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VERIFY")

# Add project root to path
project_root = "/Users/jitterx/Desktop/Aurora-X"
sys.path.append(project_root)

def test_rust_delegation():
    logger.info("--- Testing Rust Delegation ---")
    
    # 1. Kalman Filter
    from aurora_x.estimation.kalman_filter import create_kalman_filter
    kf = create_kalman_filter({}, 8, 6)
    logger.info("Kalman Filter: %s", type(kf))
    
    # 2. Physics Engine
    from aurora_x.digital_twin.physics_engine import create_physics_engine
    pe = create_physics_engine({"dt": 0.1})
    logger.info("Physics Engine: %s", type(pe))
    
    # 3. Degradation & Bayesian RUL
    from aurora_x.digital_twin.degradation_model import create_weibull_model, create_bayesian_rul
    weibull = create_weibull_model(2.5, 10000)
    bayesian = create_bayesian_rul({})
    logger.info("Weibull Model: %s", type(weibull))
    logger.info("Bayesian RUL: %s", type(bayesian))
    
    # 4. Stream Processor
    from aurora_x.pipeline.stream_processor import create_stream_processor
    sp = create_stream_processor({"window_size_samples": 100})
    logger.info("Stream Processor: %s", type(sp))
    
    # 5. Safety Controller
    from aurora_x.rl.safety_controller import SafetyController
    sc = SafetyController({})
    logger.info("Safety Controller Rust status: %s", sc._rust_sc is not None)

def test_go_ipc():
    logger.info("--- Testing Go IPC ---")
    from aurora_x.ingestion.go_client import GoServiceClient
    
    socket_path = "/tmp/aurora_go_test.sock"
    # Ensure it's clean
    if os.path.exists(socket_path):
        os.remove(socket_path)
        
    client = GoServiceClient(socket_path)
    logger.info("Client created (socket=%s)", socket_path)
    
    # Mocking Go side is hard here without another process, but we can verify it fails correctly
    logger.info("IPC Connect expected fail (no server): %s", client.connect())

if __name__ == "__main__":
    try:
        test_rust_delegation()
        test_go_ipc()
        logger.info("Verification script complete.")
    except Exception as e:
        logger.error("Verification failed: %s", e)
        sys.exit(1)
