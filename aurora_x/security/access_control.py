import logging
from typing import List, Dict, Set

logger = logging.getLogger("aurora_x.security")

class AccessTier:
    JUNIOR = "junior"
    MODERATOR = "moderator"
    SENIOR = "senior"
    MASTER = "master"

TIER_PERMISSIONS = {
    AccessTier.JUNIOR: {"development", "emergency"},
    AccessTier.MODERATOR: {"development", "staging", "maintenance", "emergency"},
    AccessTier.SENIOR: {"development", "staging", "production", "maintenance", "emergency"},
    AccessTier.MASTER: {"development", "staging", "production", "maintenance", "emergency", "key_management"}
}

# Simulated Keys
VALID_KEYS = {
    "AX-JR-1234": AccessTier.JUNIOR,
    "AX-MOD-5678": AccessTier.MODERATOR,
    "AX-SR-9012": AccessTier.SENIOR,
    "AX-MASTER-0000": AccessTier.MASTER
}

class SecurityManager:
    """Manages hierarchical access control via Software Activation Keys."""

    def __init__(self, initial_tier: str = AccessTier.JUNIOR):
        self.active_tier = initial_tier
        self.active_key = None
        logger.info("SecurityManager initialized with tier: %s", self.active_tier)

    def activate_key(self, key: str) -> bool:
        """Validate and activate a new software key."""
        if key in VALID_KEYS:
            self.active_tier = VALID_KEYS[key]
            self.active_key = key
            logger.warning("ACCESS UPGRADED: Software key activated. New Tier: %s", self.active_tier.upper())
            return True
        logger.error("INVALID KEY ATTEMPT: %s", key)
        return False

    def is_mode_allowed(self, mode: str) -> bool:
        """Check if the current tier allows switching to a specific mode."""
        allowed = TIER_PERMISSIONS.get(self.active_tier, set())
        return mode in allowed

    def can_manage_keys(self) -> bool:
        """Only Master tier can change/reset keys once activated."""
        return self.active_tier == AccessTier.MASTER or self.active_key is None

    def get_status(self) -> Dict:
        return {
            "tier": self.active_tier,
            "key_active": self.active_key is not None,
            "allowed_modes": list(TIER_PERMISSIONS.get(self.active_tier, set()))
        }
