import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    print(f"✅ Đã hoàn tất! Các hình ảnh so sánh được lưu tại thư mục: '{output_dir}/'")

if __name__ == "__main__":
    TRAIN_CSV = "p2pfl/examples/fraud/processed_data/train_processed.csv"
    TEST_CSV = "p2pfl/examples/fraud/processed_data/test_processed.csv"
    
    generate_and_save_charts(TRAIN_CSV, TEST_CSV, label_col="is_fraud")
