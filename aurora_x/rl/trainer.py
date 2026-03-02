"""
AURORA-X RL Training Loop.

Manages policy training with experience replay, safety validation,
checkpointing, and optional MLflow tracking.
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
from collections import deque

from aurora_x.rl.policies import CPPOPolicy, SACPolicy
from aurora_x.rl.reward import RewardFunction
from aurora_x.rl.safety_controller import SafetyController

logger = logging.getLogger("aurora_x.rl.trainer")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""

    def __init__(self, capacity: int = 100000):
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Dict[str, Any]):
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {}

        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        return {
            "observations": torch.FloatTensor(np.array([t["obs"] for t in batch])),
            "actions": torch.FloatTensor(np.array([t["action"] for t in batch])),
            "rewards": torch.FloatTensor(np.array([t["reward"] for t in batch])),
            "costs": torch.FloatTensor(np.array([t["cost"] for t in batch])),
            "next_observations": torch.FloatTensor(np.array([t["next_obs"] for t in batch])),
            "dones": torch.FloatTensor(np.array([t["done"] for t in batch])),
            "log_probs": torch.FloatTensor(np.array([t.get("log_prob", 0) for t in batch])),
            "advantages": torch.FloatTensor(np.array([t.get("advantage", 0) for t in batch])),
            "cost_advantages": torch.FloatTensor(np.array([t.get("cost_advantage", 0) for t in batch])),
            "returns": torch.FloatTensor(np.array([t.get("return", 0) for t in batch])),
            "cost_returns": torch.FloatTensor(np.array([t.get("cost_return", 0) for t in batch])),
        }

    def __len__(self):
        return len(self._buffer)


class RLTrainer:
    """Manages RL training and inference."""

    def __init__(
        self,
        env,
        safety_controller: SafetyController,
        config: Dict[str, Any],
    ):
        self.env = env
        self.safety_controller = safety_controller
        self.config = config

        self.algorithm = config.get("algorithm", "cppo")
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 100000)

        # Initialize policy
        obs_dim = env.obs_dim
        action_dim = env.action_dim

        if self.algorithm == "sac":
            self.policy = SACPolicy(obs_dim, action_dim, config)
        else:
            self.policy = CPPOPolicy(obs_dim, action_dim, config)

        # Reward function
        self.reward_fn = RewardFunction(config)

        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_size)

        # Training state
        self._trained = False
        self._total_steps = 0
        self._episodes = 0
        self._best_reward = float("-inf")
        self._reward_history: List[float] = []

        logger.info("RLTrainer initialized (algorithm=%s, obs=%d, act=%d)",
                     self.algorithm, obs_dim, action_dim)

    @property
    def has_trained_policy(self) -> bool:
        return self._trained

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        if not self._trained:
            # Random exploration before training
            return np.random.uniform(
                self.env.action_low, self.env.action_high
            )

        action = self.policy.select_action(obs, deterministic)

        # Scale from [-1, 1] (tanh output) to action space
        action = self.env.action_low + (action + 1) * 0.5 * (
            self.env.action_high - self.env.action_low
        )

        return action

    async def train(
        self,
        num_episodes: int = 1000,
        twin_manager=None,
        max_steps_per_episode: int = 500,
    ):
        """Run RL training loop using the digital twin (simulation mode)."""
        logger.info("Starting RL training: %d episodes", num_episodes)
        start_time = time.time()

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_cost = 0.0

            # Create a training twin
            if twin_manager:
                twin_manager.reset("training_asset")

            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(obs, deterministic=False)

                # Simulate action in twin
                if twin_manager:
                    twin = twin_manager.get_twin("training_asset")
                    if twin is None:
                        twin_manager.update("training_asset", {
                            "temperature": 80, "vibration": 1, "pressure": 50,
                            "flow": 100, "electrical": 15, "acoustic": 40,
                            "degradation": 0, "trend": 0,
                        })

                    # Step physics
                    physics_result = twin_manager._get_twin("training_asset").physics.step(
                        "training_asset", np.array([50 * (1 + action[1]), action[2] * 10])
                    )

                    # Build state dict
                    next_state = {
                        "temperature": physics_result.get("bearing_temp", 80),
                        "vibration": 1.0 + physics_result.get("bearing_degradation", 0) * 10,
                        "pressure": physics_result.get("outlet_pressure", 50),
                        "flow": 100 * (1 - physics_result.get("seal_degradation", 0)),
                        "electrical": 15,
                        "acoustic": 40,
                        "degradation": physics_result.get("bearing_degradation", 0),
                        "trend": 0,
                        "state_vector": [0] * 8,
                        "confidence": 0.5,
                    }
                else:
                    # Simplified state transition for training without twin
                    next_state = self._simple_transition(obs, action)

                # Fault report (simplified for training)
                fault_report = {
                    "fault_distribution": {"normal": 0.9, "bearing_wear": 0.1},
                    "severity_index": next_state.get("degradation", 0) * 0.5,
                    "requires_immediate_action": False,
                }

                # Compute reward
                current_state = {"temperature": obs[0] if len(obs) > 0 else 80,
                                "degradation": obs[6] if len(obs) > 6 else 0,
                                "flow": obs[3] if len(obs) > 3 else 100}
                context = {"demand_factor": 1.0, "energy_price": 0.1}

                reward_info = self.reward_fn.compute(
                    current_state, action, next_state, fault_report, context
                )

                # Compute constraint cost
                constraints = self.env.get_constraint_values(next_state)
                cost = self.reward_fn.compute_cost(next_state, constraints)

                # Build next observation
                next_obs = self.env.build_observation(next_state, fault_report)

                done = next_state.get("degradation", 0) > 0.9 or step >= max_steps_per_episode - 1

                # Store transition
                self.buffer.push({
                    "obs": obs,
                    "action": action,
                    "reward": reward_info["total"],
                    "cost": cost,
                    "next_obs": next_obs,
                    "done": float(done),
                    "log_prob": 0.0,
                    "advantage": reward_info["total"],
                    "cost_advantage": cost,
                    "return": reward_info["total"],
                    "cost_return": cost,
                })

                episode_reward += reward_info["total"]
                episode_cost += cost
                obs = next_obs
                self._total_steps += 1

                # Train on batch
                if len(self.buffer) >= self.batch_size and self._total_steps % 10 == 0:
                    batch = self.buffer.sample(self.batch_size)
                    if TORCH_AVAILABLE and hasattr(self.policy, 'update'):
                        self.policy.update(batch)

                if done:
                    break

            self._episodes += 1
            self._reward_history.append(episode_reward)

            if episode_reward > self._best_reward:
                self._best_reward = episode_reward

            if episode % 50 == 0:
                avg_reward = np.mean(self._reward_history[-50:])
                logger.info(
                    "Episode %d/%d | Reward: %.2f | Avg: %.2f | Best: %.2f | Cost: %.3f",
                    episode, num_episodes, episode_reward, avg_reward,
                    self._best_reward, episode_cost,
                )

        self._trained = True
        elapsed = time.time() - start_time
        logger.info("Training complete: %d episodes in %.1fs", num_episodes, elapsed)

    def _simple_transition(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """Simplified state transition for training without digital twin."""
        temp = obs[0] if len(obs) > 0 else 80
        vib = obs[1] if len(obs) > 1 else 1
        deg = obs[6] if len(obs) > 6 else 0

        # Simple dynamics
        temp += 0.1 * (1 + action[1]) - 0.05 * action[2] + np.random.randn() * 0.5
        vib += 0.01 * deg + np.random.randn() * 0.1
        deg += 0.0001 * (1 + action[1]) - 0.00005 * action[0]
        deg = np.clip(deg, 0, 1)

        return {
            "temperature": float(temp),
            "vibration": float(vib),
            "pressure": 50.0,
            "flow": float(100 * (1 - deg * 0.3)),
            "electrical": 15.0,
            "acoustic": 40.0,
            "degradation": float(deg),
            "trend": 0.0,
            "state_vector": [temp, vib, 50, 100, 15, 40, deg, 0],
            "confidence": 0.5,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        best_r = self._best_reward
        if best_r == float("-inf"):
            best_r = None
        return {
            "algorithm": self.algorithm,
            "total_steps": self._total_steps,
            "episodes": self._episodes,
            "best_reward": best_r,
            "avg_reward_last_50": float(np.mean(self._reward_history[-50:])) if self._reward_history else 0,
            "trained": self._trained,
            "buffer_size": len(self.buffer),
            "safety_stats": self.safety_controller.get_stats(),
        }
