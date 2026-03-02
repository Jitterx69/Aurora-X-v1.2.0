"""
AURORA-X Neural Residual Correction Network.

Small feedforward network that learns to correct Kalman filter
prediction residuals, capturing nonlinear dynamics not modeled
by the physics-based state transition.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger("aurora_x.estimation.neural_residual")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural residual correction disabled.")


if TORCH_AVAILABLE:
    class ResidualNet(nn.Module):
        """Feedforward network for residual correction."""

        def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = None):
            super().__init__()
            if output_dim is None:
                output_dim = input_dim

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
                nn.Linear(hidden_dim, output_dim),
            )


            # Small initialization for residual learning
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)


class NeuralResidualCorrector:
    """Online-trained neural network for Kalman filter residual correction.

    Architecture:
        KF prediction -> Neural correction -> Corrected prediction
        correction = ResidualNet(state_pred, measurement)
        x_corrected = x_predicted + correction
    """

    def __init__(self, config: Dict[str, Any]):
        self.hidden_dim = config.get("neural_residual", {}).get("hidden_dim", 4096)

        self.lr = config.get("neural_residual", {}).get("learning_rate", 0.001)
        self.batch_size = config.get("neural_residual", {}).get("batch_size", 32)

        self._model = None
        self._optimizer = None
        self._buffer: deque = deque(maxlen=5000)
        self._input_dim = None
        self._output_dim = None
        self._train_steps = 0
        self._enabled = TORCH_AVAILABLE

        if not self._enabled:
            logger.info("Neural residual corrector disabled (no PyTorch)")
        else:
            logger.info("Neural residual corrector ready (hidden=%d, lr=%s)",
                        self.hidden_dim, self.lr)

    def _init_model(self, input_dim: int, output_dim: int):
        """Lazy initialization once we know dimensions."""
        if not self._enabled:
            return
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._model = ResidualNet(input_dim, self.hidden_dim, output_dim)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        logger.info("ResidualNet initialized (input=%d, hidden=%d, output=%d)",
                     input_dim, self.hidden_dim, output_dim)

    def correct(
        self,
        state_pred: np.ndarray,
        measurement: np.ndarray,
    ) -> np.ndarray:
        """Apply neural correction to KF prediction.

        Args:
            state_pred: Kalman filter predicted state.
            measurement: Raw measurement vector.

        Returns:
            Corrected state estimate.
        """
        if not self._enabled or self._model is None:
            return state_pred

        # Concatenate state prediction and measurement as input
        x_input = np.concatenate([state_pred, measurement])

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_input).unsqueeze(0)
            correction = self._model(x_tensor).squeeze(0).numpy()

        # Apply residual correction (additive)
        return state_pred + correction * 0.1  # Scale factor for stability

    def observe(
        self,
        state_pred: np.ndarray,
        measurement: np.ndarray,
        state_actual: np.ndarray,
    ):
        """Store an observation for training.

        Args:
            state_pred: What KF predicted.
            measurement: What was measured.
            state_actual: What the updated state turned out to be.
        """
        if not self._enabled:
            return

        input_dim = len(state_pred) + len(measurement)
        output_dim = len(state_pred)

        if self._model is None:
            self._init_model(input_dim, output_dim)

        x_input = np.concatenate([state_pred, measurement])
        target = state_actual - state_pred  # Residual to learn

        self._buffer.append((x_input, target))

        # Train periodically
        if len(self._buffer) >= self.batch_size and self._train_steps % 10 == 0:
            self._train_step()

        self._train_steps += 1

    def _train_step(self):
        """Run one training step on the replay buffer."""
        if not self._enabled or self._model is None:
            return

        # Sample a mini-batch
        indices = np.random.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        inputs = torch.FloatTensor(np.array([b[0] for b in batch]))
        targets = torch.FloatTensor(np.array([b[1] for b in batch]))

        self._model.train()
        self._optimizer.zero_grad()

        predictions = self._model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        if self._train_steps % 100 == 0:
            logger.debug("ResidualNet loss: %.6f (step %d)", loss.item(), self._train_steps)

    @property
    def is_trained(self) -> bool:
        return self._train_steps > 100
