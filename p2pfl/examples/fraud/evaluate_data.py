import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def generate_and_save_charts(train_path, test_path, label_col="is_fraud", output_dir="p2pfl/examples/fraud/evaluate_charts"):
    # 1. Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    print("Đang tải dữ liệu...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {train_path} hoặc {test_path}!")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Biểu đồ phân phối nhãn (Class Imbalance) - So sánh Train vs Test
    print("Đang vẽ biểu đồ phân phối nhãn (Train vs Test)...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Train - Pie & Count
    train_df[label_col].value_counts().plot.pie(
        autopct='%1.2f%%', ax=axes[0, 0], cmap='Set2', explode=[0, 0.1] if train_df[label_col].nunique() == 2 else None
    )
    axes[0, 0].set_title(f'Tỷ lệ nhãn - Tập TRAIN')
    axes[0, 0].set_ylabel('')
    
    sns.countplot(x=label_col, data=train_df, ax=axes[0, 1], legend=False)
    axes[0, 1].set_title(f'Số lượng mẫu - Tập TRAIN')
    
    # Test - Pie & Count
    test_df[label_col].value_counts().plot.pie(
        autopct='%1.2f%%', ax=axes[1, 0], cmap='Pastel1', explode=[0, 0.1] if test_df[label_col].nunique() == 2 else None
    )
    axes[1, 0].set_title(f'Tỷ lệ nhãn - Tập TEST')
    axes[1, 0].set_ylabel('')
    
    sns.countplot(x=label_col, data=test_df, ax=axes[1, 1],  legend=False)
    axes[1, 1].set_title(f'Số lượng mẫu - Tập TEST')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_class_distribution_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Biểu đồ nhiệt ma trận tương quan (Correlation Heatmap) cho cả hai
    print("Đang vẽ biểu đồ ma trận tương quan...")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            plt.figure(figsize=(12, 10))
            corr_matrix = numeric_df.corr()
            mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    mask.iloc[i, j] = True
            
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", linewidths=0.5)
            plt.title(f"Ma trận tương quan - Tập {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"2_correlation_{name.lower()}.png"), dpi=300)
            plt.close()

    # 4. Biểu đồ Boxplot so sánh phân phối đặc trưng giữa Train và Test
    print("Đang vẽ biểu đồ Boxplot so sánh đặc trưng...")
    numeric_cols = [col for col in train_df.select_dtypes(include=['number']).columns if col != label_col]
    features_to_plot = numeric_cols[:4]  # Lấy 4 đặc trưng đầu tiên để minh họa
    
    if features_to_plot:
        # Chuẩn bị dữ liệu gộp để vẽ boxplot so sánh
        train_temp = train_df[features_to_plot + [label_col]].copy()
        train_temp['Dataset'] = 'Train'
        test_temp = test_df[features_to_plot + [label_col]].copy()
        test_temp['Dataset'] = 'Test'
        combined_df = pd.concat([train_temp, test_temp])

        fig, axes = plt.subplots(1, len(features_to_plot), figsize=(6 * len(features_to_plot), 6))
        if len(features_to_plot) == 1: axes = [axes]
            
        for i, col in enumerate(features_to_plot):
            sns.boxplot(x='Dataset', y=col, hue=label_col, data=combined_df, ax=axes[i], palette='Set2')
            axes[i].set_title(f'So sánh phân phối {col}')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_feature_distribution_comparison.png"), dpi=300)
        plt.close()

    # 5. Đánh giá đặc trưng bằng Mutual Information (MI)
    print("Đang đánh giá Mutual Information...")
    target_features = [
        "city_pop", "unix_time", "merch_lat", "merch_long", "distance",
        "hour", "day_of_week", "category_idx", "age", "amt_diff_avg_30d",
        "trans_count_24h", "distance_velocity", "merchant_risk_score", "merchant_freq_30d"
    ]
    
    # Lọc những cột thực sự tồn tại trong dữ liệu
    available_features = [f for f in target_features if f in train_df.columns]
    X = train_df[available_features].fillna(0)
    y = train_df[label_col]
    
    # Xác định đặc trưng rời rạc (Discrete Features) cho MI
    # Theo danh sách: hour, day_of_week, category_idx là các biến phân loại/rời rạc
    discrete_cols = ["hour", "day_of_week", "category_idx"]
    discrete_mask = [col in discrete_cols for col in available_features]
    
    # Tính toán MI score
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
    mi_series = pd.Series(mi_scores, index=available_features).sort_values(ascending=False)
    
    # Vẽ biểu đồ MI Bar Chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, palette='viridis', legend=False)
    plt.title("Mutual Information Scores (Đánh giá mức độ quan trọng của đặc trưng)")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_mutual_information_ranking.png"), dpi=300)
    plt.close()
    
    # Tự động lọc các cột có MI <= 0.001
    final_features = mi_series[mi_series > 0.001].index.tolist()
    removed_features = mi_series[mi_series <= 0.001].index.tolist()
    
    print("-" * 30)
    print(f"MI Analysis Results (Threshold > 0.001):")
    print(f"✅ Giữ lại ({len(final_features)}): {final_features}")
    if removed_features:
        print(f"❌ Loại bỏ ({len(removed_features)}): {removed_features}")
    print("-" * 30)

    print(f"✅ Đã hoàn tất! Các hình ảnh so sánh và đánh giá được lưu tại thư mục: '{output_dir}/'")

if __name__ == "__main__":
    
    # TRAIN_CSV = "p2pfl/examples/fraud/processed_data/train_processed.csv"
    # TEST_CSV = "p2pfl/examples/fraud/processed_data/test_processed.csv"
    
    # generate_and_save_charts(TRAIN_CSV, TEST_CSV, label_col="is_fraud")

    TRAIN_CSV = "p2pfl/examples/fraud/processed_data/test_preview.csv"
    TEST_CSV = "p2pfl/examples/fraud/processed_data/test_preview.csv"
    
    generate_and_save_charts(TRAIN_CSV, TEST_CSV, label_col="is_fraud")
