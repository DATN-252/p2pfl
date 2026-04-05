import os
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import kagglehub
import sys
from PIL import Image
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.getcwd())

from p2pfl.examples.aid.model.resnet_aid import ResNetAID
from p2pfl.examples.aid.transforms import get_test_transform

# --- CONFIG ---
MODEL_PATH = "best_aid_resnet50.pth" # Or path to framework saved model
DATASET_ID = "jiayuanchengala/aid-scene-classification-datasets"
OUTPUT_FILE = "aid_predictions.csv"

def run_bulk_inference():
    # 1. Dataset path
    print(f"🔍 Finding dataset {DATASET_ID}...")
    try:
        kaggle_path = kagglehub.dataset_download(DATASET_ID)
        data_dir = os.path.join(kaggle_path, "AID")
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return
    
    # 2. Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Loading model from: {MODEL_PATH} on {device}...")
    
    # We can load the LightningModule or just the raw ResNet if we only have weights
    model = ResNetAID(num_classes=30)
    try:
        # Try loading state dict (might need to handle prefix if saved by framework)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # If saved by Lightning, it might have 'model.' or 'core_model.' prefix
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ Direct load failed, trying with prefix adjustment: {e}")
        try:
            # Adjust for LightningModel wrapper
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        except Exception as e2:
             print(f"❌ Error loading model weights: {e2}")
             return

    model.to(device)
    model.eval()

    # 3. Prepare data
    transform = get_test_transform()
    classes = sorted(os.listdir(data_dir))
    
    results = []
    print("🚀 Running inference...")
    
    with torch.no_grad():
        for cls_name in tqdm(classes):
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    
                    # Transform
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Predict
                    outputs = model(img_tensor)
                    _, pred = torch.max(outputs, 1)
                    prob = torch.softmax(outputs, dim=1)[0][pred].item()
                    
                    results.append({
                        "file": img_name,
                        "true_label": cls_name,
                        "predicted_label": classes[pred.item()],
                        "confidence": prob
                    })

    # 4. Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Finished! Found {len(df)} samples. Results saved to: {OUTPUT_FILE}")
    
    # Calculate accuracy
    acc = (df["true_label"] == df["predicted_label"]).mean()
    print(f"📊 Bulk Inference Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    run_bulk_inference()
