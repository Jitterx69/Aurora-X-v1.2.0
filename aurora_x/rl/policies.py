"""
AURORA-X RL Policy Networks.

Constrained Proximal Policy Optimization (CPPO) and Soft Actor-Critic (SAC)
with Lagrangian constraint enforcement.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("aurora_x.rl.policies")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


LOG_STD_MIN = -20
LOG_STD_MAX = 2


if TORCH_AVAILABLE:
    class ActorNetwork(nn.Module):
        """Policy network that outputs action distribution."""

        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 2560):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )

            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs):
            features = self.shared(obs)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std

        def get_action(self, obs, deterministic=False):
            mean, log_std = self.forward(obs)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
                log_prob = torch.zeros(1)
            else:
                dist = Normal(mean, std)
                x = dist.rsample()
                action = torch.tanh(x)

                # Log probability with tanh correction
                log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

            return action, log_prob, mean

    class CriticNetwork(nn.Module):
        """Value network for state/state-action evaluation."""

        def __init__(self, input_dim: int, hidden_dim: int = 2560):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )


        def forward(self, x):
            return self.net(x)

    class TwinCritic(nn.Module):
        """Twin Q-networks for SAC (clipped double Q-learning)."""

        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.q1 = CriticNetwork(obs_dim + action_dim, hidden_dim)
            self.q2 = CriticNetwork(obs_dim + action_dim, hidden_dim)

        def forward(self, obs, action):
            x = torch.cat([obs, action], dim=-1)
            return self.q1(x), self.q2(x)

    class ConstraintCritic(nn.Module):
        """Cost critic for constraint estimation."""

        def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = CriticNetwork(obs_dim + action_dim, hidden_dim)

        def forward(self, obs, action):
            x = torch.cat([obs, action], dim=-1)
            return self.net(x)


class CPPOPolicy:
    """Constrained Proximal Policy Optimization with Lagrangian multipliers."""

    def __init__(self, obs_dim: int, action_dim: int, config: Dict[str, Any]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.clip_eps = 0.2
        self.constraint_threshold = config.get("constraint_threshold", 0.01)
        self.lagrangian_lr = config.get("lagrangian_lr", 0.01)

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. CPPO disabled.")
            return

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(obs_dim)
        self.cost_critic = CriticNetwork(obs_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.lr)

        # Lagrangian multiplier (learnable)
        self.log_lambda = torch.tensor(0.0, requires_grad=True)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=self.lagrangian_lr)

    @property
    def lagrangian_multiplier(self) -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        return float(torch.exp(self.log_lambda).item())

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.zeros(self.action_dim)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = self.actor.get_action(obs_tensor, deterministic)
            return action.squeeze(0).numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO update with Lagrangian constraint enforcement."""
        if not TORCH_AVAILABLE:
            return {}

        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        costs = batch["costs"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        cost_advantages = batch["cost_advantages"]
        returns = batch["returns"]
        cost_returns = batch["cost_returns"]

        # --- Update critic ---
        values = self.critic(obs).squeeze()
        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update cost critic ---
        cost_values = self.cost_critic(obs).squeeze()
        cost_critic_loss = F.mse_loss(cost_values, cost_returns)
        self.cost_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_optimizer.step()

        # --- Update actor (constrained PPO) ---
        _, new_log_probs, _ = self.actor.get_action(obs)
        # Note: simplified — in practice, need to recompute log_probs for stored actions

        ratio = torch.exp(new_log_probs.squeeze() - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        # Lagrangian penalty
        lam = torch.exp(self.log_lambda).detach()
        policy_loss = -(torch.min(surr1, surr2).mean() - lam * (cost_advantages.mean()))

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # --- Update Lagrangian multiplier ---
        avg_cost = costs.mean()
        lambda_loss = -self.log_lambda * (avg_cost - self.constraint_threshold).detach()
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "cost_critic_loss": cost_critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "lagrangian": self.lagrangian_multiplier,
            "avg_cost": avg_cost.item(),
        }


class SACPolicy:
    """Soft Actor-Critic with entropy regularization and constraint enforcement."""

    def __init__(self, obs_dim: int, action_dim: int, config: Dict[str, Any]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.tau = 0.005  # Target smoothing coefficient

        if not TORCH_AVAILABLE:
            return

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = TwinCritic(obs_dim, action_dim)
        self.critic_target = TwinCritic(obs_dim, action_dim)
        self.cost_critic = ConstraintCritic(obs_dim, action_dim)

        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(0.0, requires_grad=True)

        # Lagrangian for constraints
        self.log_lambda = torch.tensor(0.0, requires_grad=True)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=0.01)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.zeros(self.action_dim)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = self.actor.get_action(obs_tensor, deterministic)
            return action.squeeze(0).numpy()

    def soft_update_target(self):
        """Polyak averaging update of target network."""
        if not TORCH_AVAILABLE:
            return
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
