import os
import pandas as pd
import kagglehub
import transforms as transforms

def save_combined_csv(df_raw, df_processed, output_path):
    """
    Kết hợp dữ liệu gốc và các đặc trưng mới vào một file CSV.
    Loại bỏ các cột bị lặp lại (như city_pop, merch_lat,...) để file gọn hơn.
    """
    print(f"Đang kết hợp và lưu file preview tại {output_path}...")
    
    # Chỉ lấy các cột trong df_processed mà CHƯA CÓ trong df_raw (các đặc trưng mới)
    new_cols = [col for col in df_processed.columns if col not in df_raw.columns]
    df_new = df_processed[new_cols]
    
    # Ghép dữ liệu gốc với các đặc trưng mới
    combined_df = pd.concat([df_raw, df_new], axis=1)
    
    # Lưu ra CSV
    combined_df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu file preview: {output_path}")

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
    # Keep raw data for evaluation
    train_raw = train_df.copy()
    test_raw = test_df.copy()
    
    train_processed, test_processed = transforms.preprocess_transform(train_df, test_df) # type: ignore
    
    # Shuffle both raw and processed data with the same random state to keep them consistent
    print("Shuffling data...")
    train_raw_final = train_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    test_raw_final = test_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_processed_final = train_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    test_processed_final = test_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save Data
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving base CSV files to {output_dir}...")
    
    # Save Raw 
    train_raw_final.to_csv(os.path.join(output_dir, "train_raw.csv"), index=False)
    test_raw_final.to_csv(os.path.join(output_dir, "test_raw.csv"), index=False)
    
    # Save Processed
    train_processed_final.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    test_processed_final.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
    
    # Save Combined Previews (CSV instead of XLSX)
    save_combined_csv(train_raw_final, train_processed_final, os.path.join(output_dir, "train_preview.csv"))
    save_combined_csv(test_raw_final, test_processed_final, os.path.join(output_dir, "test_preview.csv"))
    
    print("✅ Pre-processing complete! (Raw, Processed, and Preview CSVs saved)")

if __name__ == "__main__":
    preprocess()
