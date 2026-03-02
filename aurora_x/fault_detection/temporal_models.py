"""
AURORA-X Temporal Deep Learning Models.

LSTM and Transformer encoder for time-series fault classification
with probabilistic outputs.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger("aurora_x.fault_detection.temporal")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Temporal models disabled.")

NUM_FAULT_CLASSES = 5


if TORCH_AVAILABLE:
    class LSTMFaultDetector(nn.Module):
        """LSTM-based temporal fault classifier."""

        def __init__(self, input_dim: int, hidden_dim: int = 800,
                     num_layers: int = 4, num_classes: int = NUM_FAULT_CLASSES):

            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=0.2,
            )
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )


        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Self-attention over LSTM outputs
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use last time step
            out = attn_out[:, -1, :]
            logits = self.fc(out)
            return logits


    class TransformerFaultDetector(nn.Module):
        """Transformer encoder for temporal fault classification."""

        def __init__(self, input_dim: int, d_model: int = 640,
                     nhead: int = 8, num_layers: int = 4,
                     num_classes: int = NUM_FAULT_CLASSES):

            super().__init__()
            self.input_projection = nn.Linear(input_dim, d_model)
            self.pos_encoding = PositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                dropout=0.2, batch_first=True,
            )

            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            encoded = self.transformer(x)
            # Global average pooling
            pooled = encoded.mean(dim=1)
            return self.classifier(pooled)


    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for Transformer."""

        def __init__(self, d_model: int, max_len: int = 500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Handle odd d_model
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


class TemporalFaultDetector:
    """Manages temporal deep learning models for fault detection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config.get("temporal", {}).get("model", "lstm")
        self.hidden_dim = config.get("temporal", {}).get("hidden_dim", 800)
        self.num_layers = config.get("temporal", {}).get("num_layers", 4)
        self.seq_length = config.get("temporal", {}).get("sequence_length", 50)


        self._model = None
        self._optimizer = None
        self._sequence_buffer: Dict[str, deque] = {}
        self._input_dim = None
        self._trained = False
        self._enabled = TORCH_AVAILABLE

        logger.info("TemporalFaultDetector initialized (model=%s, enabled=%s)",
                     self.model_type, self._enabled)

    def _init_model(self, input_dim: int):
        if not self._enabled:
            return

        self._input_dim = input_dim
        if self.model_type == "transformer":
            self._model = TransformerFaultDetector(
                input_dim, self.hidden_dim, num_layers=self.num_layers
            )
        else:
            self._model = LSTMFaultDetector(
                input_dim, self.hidden_dim, self.num_layers
            )

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        logger.info("Temporal model initialized: %s (input_dim=%d)", self.model_type, input_dim)

    def add_observation(self, asset_id: str, features: np.ndarray):
        """Add a feature vector to the asset's sequence buffer."""
        if asset_id not in self._sequence_buffer:
            self._sequence_buffer[asset_id] = deque(maxlen=self.seq_length)
        self._sequence_buffer[asset_id].append(features)

    def predict(self, asset_id: str, features: np.ndarray) -> Dict[str, Any]:
        """Predict fault using temporal context."""
        self.add_observation(asset_id, features)

        if not self._enabled:
            return self._default_prediction()

        buffer = self._sequence_buffer.get(asset_id)
        if buffer is None or len(buffer) < 10:
            return self._default_prediction()

        # Initialize model if needed
        input_dim = len(features)
        if self._model is None:
            self._init_model(input_dim)

        # Build sequence tensor
        seq = np.array(list(buffer))
        # Pad if shorter than seq_length
        if len(seq) < self.seq_length:
            padding = np.zeros((self.seq_length - len(seq), input_dim))
            seq = np.vstack([padding, seq])

        x = torch.FloatTensor(seq).unsqueeze(0)

        with torch.no_grad():
            self._model.eval()
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        fault_classes = ["normal", "bearing_wear", "misalignment", "cavitation", "overheating"]
        prob_dict = {}
        for i, cls in enumerate(fault_classes):
            if i < len(probs):
                prob_dict[cls] = float(probs[i])

        predicted_idx = int(np.argmax(probs))
        return {
            "predicted_class": predicted_idx,
            "predicted_fault": fault_classes[predicted_idx] if predicted_idx < len(fault_classes) else "unknown",
            "probabilities": prob_dict,
            "confidence": float(np.max(probs)),
            "sequence_length": len(buffer),
            "model_type": self.model_type,
        }

    def _default_prediction(self) -> Dict[str, Any]:
        return {
            "predicted_class": 0,
            "predicted_fault": "normal",
            "probabilities": {"normal": 0.8},
            "confidence": 0.0,
            "note": "insufficient_data",
        }

    @property
    def is_trained(self) -> bool:
        return self._trained
