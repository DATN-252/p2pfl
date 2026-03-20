#
# FastAPI Service for P2PFL Fraud Detection Inference (Direct Push with Persistence)
#

import os
import torch
import numpy as np
import asyncio
import io
import base64
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional
from contextlib import asynccontextmanager

# --- PATH ALERT: Update these imports when moving the file outside p2pfl ---
from p2pfl.examples.fraud.model.mlp_fraud import FraudDetectionMLP
from p2pfl.examples.fraud.transforms import haversine, calculate_age, CATEGORY_MAP
# --------------------------------------------------------------------------

# Config
CACHE_DIR = "is_model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global state
model = None
current_round = -1
model_lock = asyncio.Lock()

class ReloadRequest(BaseModel):
    model_data: str # Base64 encoded weights
    round: int
    format: str = "pt_base64"

def get_latest_from_cache():
    """Find the highest round model in the local IS cache."""
    cache_files = glob.glob(os.path.join(CACHE_DIR, "model_round_*.pt"))
    if not cache_files:
        return None, -1
    latest_file = max(cache_files, key=lambda x: int(x.split("_round_")[-1].split(".")[0]))
    round_num = int(latest_file.split("_round_")[-1].split(".")[0])
    return latest_file, round_num

async def load_model_safely(path_or_buffer, round_num: int, is_buffer=False):
    """Unified loader for both startup (file) and webhook (memory)."""
    global model, current_round
    async with model_lock:
        try:
            if round_num <= current_round:
                return

            new_model = FraudDetectionMLP(input_size=15)
            state_dict = torch.load(path_or_buffer)
            new_model.load_state_dict(state_dict)
            new_model.eval()
            
            model = new_model
            current_round = round_num
            source = "Memory" if is_buffer else "Disk Cache"
            print(f"✅ LOAD SUCCESS: Round {round_num} from {source}")
            
            # If it was a push (buffer), save it to cache for next time
            if is_buffer:
                cache_path = os.path.join(CACHE_DIR, f"model_round_{round_num}.pt")
                with open(cache_path, "wb") as f:
                    path_or_buffer.seek(0)
                    f.write(path_or_buffer.read())
                print(f"💾 PERSISTED: Model saved to local cache {cache_path}")
                
        except Exception as e:
            print(f"❌ LOAD ERROR: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Restore latest model from cache on startup."""
    cache_path, round_num = get_latest_from_cache()
    if cache_path:
        print(f"📦 Startup: Restoring latest model from cache (Round {round_num})...")
        await load_model_safely(cache_path, round_num)
    else:
        print("ℹ️ Startup: No cached model found. Waiting for first P2PFL push.")
    
    yield
    print("Inference service shutting down...")

app = FastAPI(title="P2PFL Fraud Detection Service (Persistent)", lifespan=lifespan)

class Transaction(BaseModel):
    cc_num: int
    amt: float
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    unix_time: int
    category: str
    dob: str
    trans_date_trans_time: str

@app.post("/reload")
async def trigger_reload(req: ReloadRequest, background_tasks: BackgroundTasks):
    """Receive and persist model data."""
    model_bytes = base64.b64decode(req.model_data)
    buffer = io.BytesIO(model_bytes)
    background_tasks.add_task(load_model_safely, buffer, req.round, is_buffer=True)
    return {"message": "Model received and queuing for persistence", "round": req.round}

@app.post("/predict")
async def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    
    async with model_lock:
        with torch.no_grad():
            dist = haversine(tx.lat, tx.long, tx.merch_lat, tx.merch_long)
            age = calculate_age(tx.dob)
            
            from datetime import datetime
            try:
                dt = datetime.strptime(tx.trans_date_trans_time, '%Y-%m-%d %H:%M:%S')
                hour, day_of_week = float(dt.hour), float(dt.weekday())
            except:
                hour, day_of_week = 0.0, 0.0

            category_idx = float(CATEGORY_MAP.get(tx.category, 14))
            
            features = [
                tx.amt, tx.lat, tx.long, tx.city_pop, tx.merch_lat, tx.merch_long,
                dist, hour, day_of_week, category_idx, age, float(tx.unix_time),
                0.0, 1.0, 0.0 # Behaviorals
            ]
            
            input_tensor = torch.tensor([features], dtype=torch.float32)
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()
        
    prediction = "FRAUD" if probability > 0.5 else "NORMAL"
    return {
        "fraud_probability": round(probability, 4),
        "prediction": prediction,
        "model_round": current_round
    }

@app.get("/status")
async def status():
    cache_file, _ = get_latest_from_cache()
    return {
        "status": "ready" if model else "idle", 
        "current_round": current_round,
        "cached_on_disk": cache_file
    }
