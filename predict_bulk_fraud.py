import os
import torch
import pandas as pd
import kagglehub
import sys

# Thêm thư mục hiện tại vào path để import được p2pfl
sys.path.append(os.getcwd())

from p2pfl.examples.fraud.model.mlp_fraud import FraudDetectionMLP
from p2pfl.examples.fraud.transforms import fraud_transform, build_behavioral_lookup

# --- CAU HINH ---
MODEL_PATH = "Inference_service/is_model_cache/model_round_4.pt"
DATASET_ID = "kartik2112/fraud-detection"
TEST_FILE_NAME = "fraudTest.csv"
OUTPUT_FILE = "detected_frauds.csv"

def run_bulk_inference():
    # 1. Tim duong dan tap du lieu
    print(f"🔍 Dang tim tap du lieu {DATASET_ID}...")
    try:
        kaggle_path = kagglehub.dataset_download(DATASET_ID)
        test_path = os.path.join(kaggle_path, TEST_FILE_NAME)
    except Exception as e:
        print(f"❌ Loi khi tai/tim du lieu tu Kaggle: {e}")
        return
    
    if not os.path.exists(test_path):
        print(f"❌ Khong tim thay file {TEST_FILE_NAME} tai {kaggle_path}")
        return

    # 2. Load du lieu
    print(f"📥 Dang doc du lieu tu: {test_path}...")
    df_test = pd.read_csv(test_path)
    
    # 3. Khoi tao mo hinh
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Khong tim thay file mo hinh tai: {MODEL_PATH}")
        return

    print(f"🤖 Dang load mo hinh tu: {MODEL_PATH}...")
    model = FraudDetectionMLP(input_size=15)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"❌ Loi khi load state_dict: {e}")
        return
    model.eval()

    # 4. Tien xu ly (Tinh toan behavioral features)
    print("🧠 Dang tinh toan cac dac trung hanh vi (Behavioral features)...")
    # Chuyen dataframe thanh dict format ma fraud_transform yeu cau
    examples = df_test.to_dict(orient='list')
    # Build lookup table cho behavioral features
    build_behavioral_lookup(examples)
    
    # Transform toan bo tap test
    print("✨ Dang chuan hoa du lieu...")
    transformed_data = fraud_transform(examples)
    features = torch.stack(transformed_data["features"])

    # 5. Chay du doan
    print("🚀 Dang chay du doan tren toan bo tap du lieu...")
    with torch.no_grad():
        logits = model(features)
        # Handle single output or batch output
        if len(logits.shape) > 1:
            probabilities = torch.sigmoid(logits).squeeze().numpy()
        else:
            probabilities = torch.sigmoid(logits).numpy()
    
    # Them ket qua vao dataframe goc
    df_test['fraud_probability'] = probabilities
    df_test['is_predicted_fraud'] = (probabilities > 0.5).astype(int)

    # 6. Loc cac mau gian lan
    fraud_samples = df_test[df_test['is_predicted_fraud'] == 1]
    
    print(f"✅ Hoan tat! Tim thay {len(fraud_samples)} mau nghi van gian lan.")
    fraud_samples.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Da luu ket qua vao file: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_bulk_inference()
