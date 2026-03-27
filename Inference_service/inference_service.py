#
# FastAPI Service for P2PFL Fraud Detection Inference (Enterprise Edition)
# Features: Versioning, Rollback, and Trigger Control
#

import os
import torch
import numpy as np
import asyncio
import io
import base64
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# --- PATH ALERT: Update these imports when moving the file outside p2pfl ---
from p2pfl.examples.fraud.model.mlp_fraud import FraudDetectionMLP
from p2pfl.examples.fraud.transforms import haversine, calculate_age, CATEGORY_MAP
# --------------------------------------------------------------------------

# Config
CACHE_DIR = "is_model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global State
model = None
current_round = -1
accept_triggers = True # Global switch for FL triggers
model_lock = asyncio.Lock()

class ReloadRequest(BaseModel):
    model_data: str 
    round: int
    format: str = "pt_base64"

class ConfigRequest(BaseModel):
    accept_triggers: bool

def get_available_versions() -> List[int]:
    """List all round numbers available in the local cache."""
    files = glob.glob(os.path.join(CACHE_DIR, "model_round_*.pt"))
    rounds = [int(f.split("_round_")[-1].split(".")[0]) for f in files]
    return sorted(rounds, reverse=True)

async def load_model_logic(path_or_buffer, round_num: int, is_buffer=False):
    """Core logic to swap the active model."""
    global model, current_round
    async with model_lock:
        try:
            new_model = FraudDetectionMLP(input_size=15)
            # Load weights
            state_dict = torch.load(path_or_buffer)
            new_model.load_state_dict(state_dict)
            new_model.eval()
            
            # Atomic swap
            model = new_model
            current_round = round_num
            
            # Persist if it came from memory
            if is_buffer:
                cache_path = os.path.abspath(os.path.join(CACHE_DIR, f"model_round_{round_num}.pt"))
                with open(cache_path, "wb") as f:
                    path_or_buffer.seek(0)
                    f.write(path_or_buffer.read())
                print(f"💾 PERSISTED: Model saved to {cache_path}")
            
            source = "MEMORY" if is_buffer else "DISK"
            print(f"✅ ACTIVE MODEL UPDATED: Round {round_num} from {source}")
            return True
        except Exception as e:
            print(f"❌ LOAD ERROR: {e}")
            return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Restore the latest version on startup."""
    versions = get_available_versions()
    if versions:
        latest_r = versions[0]
        path = os.path.abspath(os.path.join(CACHE_DIR, f"model_round_{latest_r}.pt"))
        print(f"📦 Startup: Restoring Round {latest_r} from {path}...")
        await load_model_logic(path, latest_r)
    else:
        print("ℹ️ Startup: No cached model found in is_model_cache/")
    yield
    print("Shutting down...")

app = FastAPI(title="P2PFL Enterprise Inference Service", lifespan=lifespan)

# --- ADMIN / CONTROL ENDPOINTS ---

@app.get("/admin/status")
async def get_status():
    return {
        "status": "ready" if model else "idle",
        "current_round": current_round,
        "accept_triggers": accept_triggers,
        "available_versions": get_available_versions(),
        "cache_dir": os.path.abspath(CACHE_DIR)
    }

@app.post("/admin/config")
async def update_config(req: ConfigRequest):
    global accept_triggers
    accept_triggers = req.accept_triggers
    status = "ENABLED" if accept_triggers else "DISABLED"
    print(f"⚙️ ADMIN: Trigger reception is now {status}")
    return {"message": f"Trigger reception {status.lower()}", "accept_triggers": accept_triggers}

@app.post("/admin/rollback/{round_num}")
async def rollback(round_num: int):
    """Manually switch to a specific version in cache."""
    cache_path = os.path.join(CACHE_DIR, f"model_round_{round_num}.pt")
    if not os.path.exists(cache_path):
        raise HTTPException(status_code=404, detail=f"Version {round_num} not found in cache")
    
    success = await load_model_logic(cache_path, round_num)
    if success:
        return {"message": "Rollback successful", "active_round": round_num}
    raise HTTPException(status_code=500, detail="Failed to load model during rollback")

# --- P2PFL TRIGGER ENDPOINT ---

@app.post("/reload")
async def trigger_reload(req: ReloadRequest, background_tasks: BackgroundTasks):
    """Triggered by P2PFL ModelPackager."""
    if not accept_triggers:
        print(f"🚫 TRIGGER BLOCKED: Incoming push for Round {req.round} ignored")
        raise HTTPException(status_code=403, detail="Inference Service is currently not accepting automated triggers.")
    
    if req.round <= current_round:
        print(f"ℹ️ PUSH IGNORED: Round {req.round} is not newer than current Round {current_round}")
        return {"message": "Already up to date", "current": current_round}

    print(f"📥 PUSH RECEIVED: Incoming Model Round {req.round}")
    model_bytes = base64.b64decode(req.model_data)
    buffer = io.BytesIO(model_bytes)
    background_tasks.add_task(load_model_logic, buffer, req.round, is_buffer=True)
    return {"message": "Push accepted", "target_round": req.round}

# --- PREDICTION ENDPOINT ---

from p2pfl.examples.fraud.transforms import fraud_transform, build_behavioral_lookup

@app.post("/predict")
async def predict(tx: Dict = Body(...)): 
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    
    try:
        async with model_lock:
            with torch.no_grad():
                example = {k: [v] for k, v in tx.items()}
                
                build_behavioral_lookup(example)
                
                transformed = fraud_transform(example)
                features = torch.stack(transformed["features"])  
                logits = model(features)
                probability = torch.sigmoid(logits).item()
        
        prediction = "FRAUD" if probability > 0.5 else "NORMAL"
        return {
            "fraud_probability": round(probability, 4),
            "prediction": prediction,
            "model_round": current_round
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid transaction data: {e}")
