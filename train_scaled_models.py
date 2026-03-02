import os
import sys
import torch
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TRAIN")

# Add project root to path
project_root = os.getcwd()
sys.path.append(project_root)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ResidualDataset(Dataset):
    def __init__(self, df):
        self.inputs = torch.FloatTensor(df.iloc[:, :14].values)
        self.targets = torch.FloatTensor(df.iloc[:, 14:22].values)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

def load_parquet_data(name: str):
    path = f"aurora_x/ml_data/datasets/{name}.parquet"
    if not os.path.exists(path):
        logger.error(f"Dataset {path} not found.")
        return None
    return pd.read_parquet(path)

def train_residual_corrector():
    from aurora_x.estimation.neural_residual import NeuralResidualCorrector
    logger.info("--- Deep Training: NeuralResidualCorrector ---")
    
    df = load_parquet_data("estimation_data")
    if df is None: return {}
    
    split = int(len(df) * 0.8)
    train_ds = ResidualDataset(df.iloc[:split])
    test_ds = ResidualDataset(df.iloc[split:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    corrector = NeuralResidualCorrector({"neural_residual": {"hidden_dim": 4096, "learning_rate": 1e-4}})
    corrector._init_model(14, 8)
    model = corrector._model
    optimizer = corrector._optimizer
    
    # Target: RMSE < 0.1
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            test_preds = model(test_ds.inputs)
            rmse = torch.sqrt(F.mse_loss(test_preds, test_ds.targets)).item()
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f} | Test RMSE: {rmse:.6f}")
        if rmse < 0.1:
            logger.info("Target RMSE (<0.1) achieved!")
            break

    return {
        "model": "ResidualNet",
        "params": count_parameters(model),
        "rmse": float(rmse),
        "status": "CONVERGED" if rmse < 0.1 else "TRAINED"
    }

class FaultDataset(Dataset):
    def __init__(self, df):
        self.inputs = torch.FloatTensor(df.iloc[:, :1000].values)
        self.targets = torch.LongTensor(df['target_class'].values)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

def train_temporal_detector():
    from aurora_x.fault_detection.temporal_models import TemporalFaultDetector
    logger.info("--- Deep Training: Temporal Detectors ---")
    
    df = load_parquet_data("fault_data")
    if df is None: return []
    
    split = int(len(df) * 0.8)
    train_loader = DataLoader(FaultDataset(df.iloc[:split]), batch_size=64, shuffle=True)
    test_ds = FaultDataset(df.iloc[split:])

    results = []
    for m_type in ["lstm", "transformer"]:
        hidden = 800 if m_type == "lstm" else 640
        # Reset to seq_len=1 for maximum signal clarity
        config = {"temporal": {"model": m_type, "hidden_dim": hidden, "num_layers": 4, "sequence_length": 1}}
        detector = TemporalFaultDetector(config)
        detector._init_model(1000)
        
        model = detector._model
        lr = 1e-4 if m_type == "transformer" else 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        epochs = 20
        for epoch in range(epochs):
            model.train()
            correct_train = 0
            for x, y in train_loader:
                # Add seq dim: (batch, seq=1, feat=1000)
                x_seq = x.unsqueeze(1)
                optimizer.zero_grad()
                logits = model(x_seq)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                
                pred_idx = logits.argmax(dim=1)
                correct_train += (pred_idx == y).sum().item()
            
            # Eval
            model.eval()
            with torch.no_grad():
                x_test = test_ds.inputs.unsqueeze(1)
                test_logits = model(x_test)
                test_preds = test_logits.argmax(dim=1)
                accuracy = (test_preds == test_ds.targets).sum().item() / len(test_ds)
            
            logger.info(f"[{m_type.upper()}] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Test Acc: {accuracy:.4f}")
            if accuracy > 0.97:
                logger.info(f"Target Accuracy (>97%) achieved for {m_type}!")
                break
        
        results.append({
            "model": m_type.upper(),
            "params": count_parameters(model),
            "accuracy": float(accuracy),
            "status": "OPTIMIZED" if accuracy > 0.97 else "TRAINED"
        })
        
    return results


def train_rl_policy():
    from aurora_x.rl.policies import CPPOPolicy
    logger.info("--- Scaling & Training RL Policy ---")
    policy = CPPOPolicy(20, 4, {"hidden_dim": 2560})
    return {
        "model": "RL CPPO",
        "actor_params": count_parameters(policy.actor),
        "critic_params": count_parameters(policy.critic),
        "status": "INITIALIZED"
    }

if __name__ == "__main__":
    import json
    results = {}
    try:
        results["residual"] = train_residual_corrector()
        results["temporal"] = train_temporal_detector()
        results["rl"] = train_rl_policy()
        
        print("\n--- HYPER-OPTIMIZATION REPORT (97%+ TARGET) ---")
        print(json.dumps(results, indent=2))
        print("-----------------------------------------------\n")
    except Exception as e:
        logger.error("Hyper-optimization failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)




