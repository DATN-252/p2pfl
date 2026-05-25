import os
import torch
import numpy as np
import asyncio
import io
import base64
import glob
import pandas as pd
import kagglehub
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# --- PATH ALERT ---
from p2pfl.examples.fraud.model.mlp_fraud import FraudDetectionMLP
from p2pfl.examples.fraud.transforms import fraud_transform, build_behavioral_lookup

# Config
# Đảm bảo CACHE_DIR trỏ đúng vào thư mục is_model_cache bên trong Inference_service
# cho dù bạn chạy uvicorn từ thư mục gốc hay thư mục Inference_service.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "is_model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DATASET_ID = "kartik2112/fraud-detection"
TEST_FILE_NAME = "fraudTest.csv"

# Global State
model = None
current_round = -1
accept_triggers = True
model_lock = asyncio.Lock()
# Biến để lưu trữ lookup data nhằm đảm bảo tính nhất quán
global_lookup_data = None

class ReloadRequest(BaseModel):
    model_data: str
    round: int
    format: str = "pt_base64"

class ConfigRequest(BaseModel):
    accept_triggers: bool

def get_available_versions() -> List[int]:
    files = glob.glob(os.path.join(CACHE_DIR, "model_round_*.pt"))
    rounds = [int(f.split("_round_")[-1].split(".")[0]) for f in files]
    return sorted(rounds, reverse=True)

async def load_model_logic(path_or_buffer, round_num: int, is_buffer=False):
    global model, current_round
    async with model_lock:
        try:
            # 0. Check if file is empty (prevents EOFError/invalid load key)
            if not is_buffer and os.path.exists(path_or_buffer) and os.path.getsize(path_or_buffer) == 0:
                print(f"⚠️ SKIPPING: Model file {path_or_buffer} is empty (0 bytes).")
                return False

            new_model = FraudDetectionMLP(input_size=23)
            
            # 1. Load data from path or buffer
            data = torch.load(path_or_buffer, map_location="cpu")
            
            # 2. Robust state_dict extraction
            # Handles: pure state_dict, full model object, or dict with "state_dict" key
            state_dict = None
            if isinstance(data, dict):
                state_dict = data.get("state_dict", data)
            elif hasattr(data, "state_dict"):
                state_dict = data.state_dict()
            else:
                state_dict = data # Fallback
            
            new_model.load_state_dict(state_dict)
            new_model.eval()

            model = new_model
            current_round = round_num

            # 3. Save to cache if it was pushed via buffer
            if is_buffer:
                cache_path = os.path.abspath(os.path.join(CACHE_DIR, f"model_round_{round_num}.pt"))
                # Use the original bytes from the buffer (getvalue() is safer than read() after torch.load)
                model_bytes = path_or_buffer.getvalue()
                with open(cache_path, "wb") as f:
                    f.write(model_bytes)
                print(f"💾 Model Round {round_num} cached to {cache_path}")
            
            print(f"✅ LOAD SUCCESS: Model Round {round_num} is now active.")
            return True
        except Exception as e:
            print(f"❌ LOAD ERROR (Round {round_num}): {e}")
            return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
    1. Restore model mới nhất.
    2. Download & Load dataset từ Kaggle để giả lập Database Lookup.
    """
    global global_lookup_data

    # 1. Khôi phục Model
    versions = get_available_versions()
    if versions:
        latest_r = versions[0]
        path = os.path.abspath(os.path.join(CACHE_DIR, f"model_round_{latest_r}.pt"))
        print(f"📦 Startup: Đang load Model Round {latest_r}...")
        await load_model_logic(path, latest_r)

    # 2. Download Dataset để làm Mock Database
    print(f"🔍 Startup: Đang tải dataset {DATASET_ID} từ Kaggle...")
    try:
        tmp_path = kagglehub.dataset_download(DATASET_ID)
        csv_path = os.path.join(tmp_path, TEST_FILE_NAME)
        print(f"📥 Startup: Đang nạp dữ liệu từ {csv_path} vào bộ nhớ...")

        # Load CSV và chuyển thành format dict của p2pfl (giống predict_bulk_fraud.py)
        df = pd.read_csv(csv_path)
        global_lookup_data = df.to_dict(orient='list')

        # Khởi tạo lookup table global một lần duy nhất
        build_behavioral_lookup(global_lookup_data)

        # MỚI: Khởi tạo FEATURE_STATS (mean/std) bằng cách chạy transform trên toàn bộ dataset
        # Nếu không có bước này, request đầu tiên (single transaction) sẽ khiến FEATURE_STATS = [0,0,0...]  
        print("✨ Startup: Đang khởi tạo Feature Statistics (Standardization)...")
        fraud_transform(global_lookup_data)

        print("✅ Startup: Hệ thống Lookup và Feature Stats đã sẵn sàng.")
    except Exception as e:
        print(f"⚠️ Startup Warning: Không thể khởi tạo database lookup: {e}")

    yield
    print("Shutting down...")

app = FastAPI(title="P2PFL Enterprise Inference Service", lifespan=lifespan)

# --- ENDPOINTS ---

@app.post("/predict")
async def predict(tx: Dict = Body(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    try:
        # Sử dụng Read-Copy-Update pattern nhẹ để tăng concurrency
        local_model = model

        # Chuyển transaction đơn lẻ thành format batch (list) để tương thích với transform
        example = {k: [v] for k, v in tx.items()}

        # Lưu ý: Vì build_behavioral_lookup đã được gọi ở lifespan trên toàn bộ dataset,
        # nên các biến global trong module transforms đã được populate.
        # Chúng ta gọi lại cho example hiện tại để cập nhật/trích xuất đặc trưng.
        build_behavioral_lookup(example)

        transformed = fraud_transform(example)
        features = torch.stack(transformed["features"])

        with torch.no_grad():
            logits = local_model(features)
            probability = torch.sigmoid(logits).item()

        prediction = "FRAUD" if probability > 0.5 else "NORMAL"
        return {
            "fraud_probability": round(probability, 4),
            "prediction": prediction,
            "model_round": current_round,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu: {e}")

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
