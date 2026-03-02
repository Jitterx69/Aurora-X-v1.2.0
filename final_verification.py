import asyncio
import logging
import sys
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VERIFY_INTEGRATED")

# Add project root to path
project_root = os.getcwd()
sys.path.append(project_root)

async def test_integrated_secure_pipeline():
    from aurora_x.main import AuroraXPlatform
    
    logger.info("Initializing AuroraXPlatform for secure pipeline test...")
    # Minimal config for test
    platform = AuroraXPlatform()
    # Disable prometheus to avoid port conflict
    platform.config.section("observability")["prometheus"]["enabled"] = False
    
    # We only need to test the _secure_process method directly to avoid starting all servers
    await asyncio.wait_for(platform.setup(), timeout=180)

    
    # Mock event
    event = {
        "asset_id": "test-asset-001",
        "timestamp": 123456789.0,
        "sensors": {
            "temperature": 85.0,
            "vibration": 2.5,
            "pressure": 60.0,
            "flow": 100.0,
            "electrical": 15.0,
            "acoustic": 40.0
        }
    }
    
    logger.info("Triggering _secure_process...")
    result = await platform._secure_process("test-asset-001", event)
    
    if result and result.get("secure_status") == "verified":
        logger.info("✓ Secure pipeline verification successful!")
        logger.info(f"Faderated State Keys: {result.get('state', {}).keys()}")
        logger.info(f"Fault Report Severity: {result.get('fault_report', {}).get('severity_index')}")
        logger.info(f"Safe Action: {result.get('safe_action')}")
    else:
        logger.error("✗ Secure pipeline verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(test_integrated_secure_pipeline())
    except Exception as e:
        logger.error(f"Verification crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
