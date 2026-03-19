import os
import pandas as pd
import kagglehub
import p2pfl.examples.fraud.transforms as transforms

def preprocess(dataset_id="kartik2112/fraud-detection", output_dir="p2pfl/examples/fraud/processed_data"):
    # Download and Load Data
    print(f"Downloading dataset {dataset_id}...")
    path = kagglehub.dataset_download(dataset_id)
    train_path = os.path.join(path, "fraudTrain.csv")
    test_path = os.path.join(path, "fraudTest.csv")
    
    print("Loading CSV files...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Feature Engineering
    print("Performing feature engineering ...")
    train_df, test_df = transforms.preprocess_transform(train_df, test_df)
    
    # Shuffle data trực tiếp (không cần tách X, y nếu không dùng SMOTE)
    train_final = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_final = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save Processed 
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train_processed.csv")
    test_output = os.path.join(output_dir, "test_processed.csv")
    
    print(f"Saving processed data to {output_dir}...")
    train_final.to_csv(train_output, index=False)
    test_final.to_csv(test_output, index=False)
    
    print("✅ Pre-processing complete!")

if __name__ == "__main__":
    preprocess()