
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from p2pfl.examples.fraud.model.mlp_fraud import model_build_fn
from p2pfl.utils.model_packaging import ModelPackager
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from datetime import datetime
# Configuration
TRAIN_PATH = "p2pfl/examples/fraud/processed_data/train_processed.csv"
TEST_PATH = "p2pfl/examples/fraud/processed_data/test_processed.csv"
ROUNDS = 100
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
ALPHA = 0.95
GAMMA = 0.5
PACKAGING_INTERVAL = 25
OUTPUT_DIR = "models/fraud_onnx_manual"

class FraudCSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Assuming NUMERIC_FEATURES (23 features) as per transforms.py
        # One-hot encoded cats are included in the CSV columns
        self.y = torch.tensor(df["is_fraud"].values, dtype=torch.float32).unsqueeze(1)
        self.X = torch.tensor(df.drop("is_fraud", axis=1).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {"features": self.X[idx], "label": self.y[idx]}

def train_manual():
    now = datetime.now()
    formatted_time = now.strftime("%d-%m-%Y--%H-%M")
    CUR_OUTPUT_DIR = OUTPUT_DIR + "--" + formatted_time
    os.makedirs(CUR_OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"📂 Loading data from {TRAIN_PATH}...")
    train_ds = FraudCSVDataset(TRAIN_PATH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model (using the P2PFL wrapper build function)
    print("🤖 Initializing MLP model...")
    p2pfl_model = model_build_fn(
        input_size=23, 
        learning_rate=LEARNING_RATE, 
        alpha=ALPHA, 
        gamma=GAMMA
    )
    model = p2pfl_model.get_model()
    optimizer = model.configure_optimizers()
    
    # 3. Initialize Packager
    packager = ModelPackager()
    
    print(f"🚀 Starting training loop for {ROUNDS} rounds...")
    for round_num in range(1, ROUNDS + 1):
        model.train()
        total_loss = 0
        
        # In this manual script, 1 round = 1 epoch of training
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Round {round_num} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")
        
        avg_loss = total_loss / len(train_loader)
        print(f"\n✅ Round {round_num} complete. Average Loss: {avg_loss:.6f}")
        
        # 4. Packaging Logic (Save & Export ONNX)
        if round_num % PACKAGING_INTERVAL == 0 or round_num == ROUNDS:
            print(f"📦 Packaging and Exporting ONNX for Round {round_num}...")
            # We call the internal _save_model_locally which now includes ONNX export
            packager._save_model_locally(p2pfl_model, round_num, CUR_OUTPUT_DIR)

    print(f"\n✨ Training Finished! Models saved in: {os.path.abspath(CUR_OUTPUT_DIR)}")

if __name__ == "__main__":
    train_manual()
