import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def generate_and_save_charts(train_path, test_path, label_col="is_fraud", output_dir="p2pfl/examples/fraud/evaluate_charts"):
    # 1. Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Thiết lập kích thước chữ chung
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})

    print("Đang tải dữ liệu...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {train_path} hoặc {test_path}!")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
   
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
        # Loại bỏ is_fraud khỏi ma trận tương quan
        numeric_df = df.select_dtypes(include=['number']).drop(columns=[label_col], errors='ignore')
        if not numeric_df.empty:
            plt.figure(figsize=(12, 10))
            corr_matrix = numeric_df.corr()
            mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    mask.iloc[i, j] = True
            
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", linewidths=0.5)
            # plt.title(f"Ma trận tương quan - Tập {name} (Không bao gồm {label_col})", fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"2_correlation_{name.lower()}.png"), dpi=300)
            plt.close()

    # 4. Biểu đồ Boxplot so sánh phân phối đặc trưng giữa Train và Test
    print("Đang vẽ biểu đồ Boxplot so sánh đặc trưng...")
    numeric_cols = [col for col in train_df.select_dtypes(include=['number']).columns if col != label_col]
    # Ưu tiên các biến quan trọng để đánh giá việc chọn dữ liệu
    priority_features = ["amt", "log_amt", "amt_zscore", "distance_velocity", "age"]
    features_to_plot = [f for f in priority_features if f in numeric_cols]
    if not features_to_plot:
        features_to_plot = numeric_cols[:4]
    
    if features_to_plot:
        # Chuẩn bị dữ liệu gộp để vẽ boxplot so sánh
        train_temp = train_df[features_to_plot].copy()
        train_temp['Dataset'] = 'Train'
        test_temp = test_df[features_to_plot].copy()
        test_temp['Dataset'] = 'Test'
        combined_df = pd.concat([train_temp, test_temp])

        fig, axes = plt.subplots(1, len(features_to_plot), figsize=(6 * len(features_to_plot), 6))
        if len(features_to_plot) == 1: axes = [axes]
            
        for i, col in enumerate(features_to_plot):
            # Loại bỏ hue=label_col để chỉ so sánh phân phối giữa các tập dữ liệu
            # Cập nhật theo khuyến nghị của Seaborn: gán hue='Dataset' và legend=False
            sns.boxplot(x='Dataset', y=col, hue='Dataset', data=combined_df, ax=axes[i], palette='Set2', legend=False)
            # axes[i].set_title(f'So sánh {col}', fontsize=20, fontweight='bold')
            axes[i].set_xlabel('Tập dữ liệu', fontsize=16)
            axes[i].set_ylabel(col, fontsize=16)
            axes[i].tick_params(labelsize=14)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_feature_distribution_comparison.png"), dpi=300)
        plt.close()


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
    discrete_cols = ["hour", "day_of_week", "category_idx"]
    discrete_mask = [col in discrete_cols for col in available_features]
    
    # Tính toán MI score
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
    mi_series = pd.Series(mi_scores, index=available_features).sort_values(ascending=False)
    
    # Vẽ biểu đồ MI Bar Chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, palette='viridis', legend=False)
    # plt.title("Mutual Information Scores (Đánh giá mức độ quan trọng của đặc trưng)")
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_mutual_information_ranking.png"), dpi=300)
    plt.close()


    print(f"✅ Đã hoàn tất! Các hình ảnh so sánh và đánh giá được lưu tại thư mục: '{output_dir}/'")

if __name__ == "__main__":
    
    TRAIN_CSV = "p2pfl/examples/fraud/processed_data/train_preview.csv"
    TEST_CSV = "p2pfl/examples/fraud/processed_data/test_preview.csv"
    
    generate_and_save_charts(TRAIN_CSV, TEST_CSV, label_col="is_fraud")
