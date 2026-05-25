import onnx
import onnxruntime as ort
import numpy as np
import os
import json

def verify_model():
    model_path = r"models\fraud_onnx_manual--25-05-2026--03-41\packaged_model_round_100.onnx"
    json_path = "sample_input.json"
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"❌ JSON input file not found: {json_path}")
        return

    print(f"🚀 Verifying model: {model_path}")
    print(f"📄 Loading input data from: {json_path}")
    
    try:
        # 1. Load input data from JSON
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Ensure correct order of 23 features
        feature_order = [
            "city_pop", "hour", "age", "unix_time",
            "amt_diff_avg_30d", "trans_count_24h", "distance_velocity", 
            "merchant_risk_score", "merchant_freq_30d",
            "cat_misc_net", "cat_grocery_pos", "cat_entertainment", "cat_gas_transport",
            "cat_misc_pos", "cat_grocery_net", "cat_shopping_net", "cat_shopping_pos",
            "cat_food_dining", "cat_personal_care", "cat_health_fitness", "cat_travel",
            "cat_kids_pets", "cat_home"
        ]
        
        input_values = [float(data[feat]) for feat in feature_order]
        input_array = np.array([input_values]).astype(np.float32)

        # 2. Load and check ONNX structure
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX Structure: VALID")

        # 3. Run Inference with ONNX Runtime
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        outputs = session.run(None, {input_name: input_array})
        logits = outputs[0][0][0]
        
        # Apply Sigmoid to get probability (since raw output is Logits)
        probability = 1 / (1 + np.exp(-logits))
        
        print("\n" + "="*40)
        print("           PREDICTION RESULTS")
        print("="*40)
        print(f"Logits Output: {logits:.4f}")
        print(f"Fraud Probability: {probability:.4f}")
        print(f"Prediction: {'🔴 FRAUD' if probability > 0.5 else '🟢 NORMAL'}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    verify_model()

if __name__ == "__main__":
    verify_model()
