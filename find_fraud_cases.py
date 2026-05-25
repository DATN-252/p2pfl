import onnxruntime as ort
import numpy as np
import pandas as pd
import os

def find_fraud_cases():
    model_path = r"models\fraud_onnx_manual--25-05-2026--03-41\packaged_model_round_100.onnx"
    csv_path = r"p2pfl\examples\fraud\processed_data\test_processed.csv"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    print(f"🚀 Loading model: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    print(f"📄 Reading data from: {csv_path}")
    # Load in chunks if the file is too large, but for now let's try reading it all if it fits.
    # The file is ~148MB, which should fit in memory.
    df = pd.read_csv(csv_path)
    
    # Feature columns (first 23 columns)
    feature_cols = df.columns[:23].tolist()
    print(f"✅ Using {len(feature_cols)} features for inference.")
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Run inference in batch for efficiency
    print("🧠 Running inference on all rows...")
    # Some ONNX models might have issues with very large batches, let's do it in chunks of 10000
    batch_size = 10000
    all_probs = []
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        # X_batch shape is (N, 23). Model expects (N, 1, 23) or (N, 23)? 
        # Looking at verify_model_100.py: input_array = np.array([input_values]).astype(np.float32)
        # input_array shape is (1, 23). 
        # Let's check session input shape
        outputs = session.run(None, {input_name: X_batch})
        logits = outputs[0]
        # Probability calculation (Sigmoid)
        probs = 1 / (1 + np.exp(-logits))
        all_probs.extend(probs.flatten())
        
    df['predicted_prob'] = all_probs
    df['prediction'] = (df['predicted_prob'] > 0.5).astype(int)
    
    fraud_cases = df[df['prediction'] == 1]
    
    print(f"\nTotal rows processed: {len(df)}")
    print(f"Total fraud cases found: {len(fraud_cases)}")
    
    if len(fraud_cases) > 0:
        output_file = "found_fraud_cases.csv"
        fraud_cases.to_csv(output_file, index=False)
        print(f"✅ Fraud cases saved to: {output_file}")
        
        # Show a few examples
        print("\n--- Example Fraud Cases ---")
        print(fraud_cases.head())
    else:
        print("No fraud cases found with the current threshold.")

if __name__ == "__main__":
    find_fraud_cases()
