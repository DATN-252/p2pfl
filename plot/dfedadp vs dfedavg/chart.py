import re
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

LOG_CONFIG = {
    'DFedAdp': [
        'DFedAdp',
        'DFedAdp.1',
    ],
    'FedAvg': [
        'FedAvg',
        'FedAvg.1',
    ]
}

def parse_single_file(filename):
    data_list = []
    if not os.path.exists(filename):
        # In nhẹ thông báo nếu thiếu file, nhưng không dừng chương trình
        print(f"⚠️  File not found (skipped): {filename}")
        return []

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    log_pattern = re.compile(r'\[ (.*?) \| (.*?) \| INFO \] 📈 Evaluated. Results: (\{.*\})')
    
    for line in lines:
        match = log_pattern.search(line)
        if match:
            timestamp_str = match.group(1)
            node_id = match.group(2)
            json_str = match.group(3)
            try:
                metrics = ast.literal_eval(json_str)
                clean_metrics = {k.strip(): v for k, v in metrics.items()}
                entry = {
                    'timestamp': pd.to_datetime(timestamp_str),
                    'node_id': node_id,
                    'accuracy': clean_metrics.get('test_acc'),
                    'loss': clean_metrics.get('test_loss')
                }
                data_list.append(entry)
            except Exception:
                continue
    return data_list

def process_experiment_group(file_list):
    all_records = []
    for filename in file_list:
        all_records.extend(parse_single_file(filename))
        
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Sắp xếp theo thời gian để tính Round chuẩn xác
    df = df.sort_values(by=['timestamp']) 
    df['round'] = df.groupby('node_id').cumcount() + 1
    return df

def main():
    final_summary_list = [] # Chứa dữ liệu trung bình để so sánh
    all_raw_data = []       # Chứa dữ liệu chi tiết từng node để vẽ biểu đồ riêng

    print("🔄 Processing logs...")

    for label, files in LOG_CONFIG.items():
        df_group = process_experiment_group(files)
        
        if not df_group.empty:
            df_group['run_label'] = label
            all_raw_data.append(df_group)

            # Tính trung bình cho biểu đồ tổng hợp
            summary = df_group.groupby('round').agg({
                'accuracy': 'mean',
                'loss': 'mean'
            }).reset_index()
            summary['run_label'] = label
            final_summary_list.append(summary)
    
    if not final_summary_list:
        print("❌ No valid data found.")
        return

    # --- BIỂU ĐỒ 1: SO SÁNH TRUNG BÌNH (AVERAGE COMPARISON) ---
    print("📊 Plotting Average Comparison...")
    all_summary = pd.concat(final_summary_list)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for label in all_summary['run_label'].unique():
        subset = all_summary[all_summary['run_label'] == label]
        ax1.plot(subset['round'], subset['accuracy'], label=label, linewidth=2.5)
        ax2.plot(subset['round'], subset['loss'], label=label, linewidth=2.5)
    
    ax1.set_title('Average Test Accuracy')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Accuracy')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.set_title('Average Test Loss')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()
    
    plt.tight_layout()
    plt.savefig('chart_1_comparison_avg.png')
    plt.show()

    # --- BIỂU ĐỒ 2+: CHI TIẾT TỪNG NODE CHO MỖI GIẢI THUẬT ---
    # Vẽ riêng mỗi giải thuật một hình để tránh rối (Spaghetti Plot)
    for df_algo in all_raw_data:
        algo_name = df_algo['run_label'].iloc[0]
        print(f"📊 Plotting Node Details for: {algo_name}...")
        
        fig_detail, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Lấy danh sách các node
        nodes = df_algo['node_id'].unique()
        
        # Vẽ đường cho từng node (mờ hơn để nhìn thấy mật độ)
        for node in nodes:
            node_data = df_algo[df_algo['node_id'] == node]
            ax_d1.plot(node_data['round'], node_data['accuracy'], color='blue', alpha=0.15, linewidth=1)
            ax_d2.plot(node_data['round'], node_data['loss'], color='red', alpha=0.15, linewidth=1)

        # Vẽ thêm đường trung bình đậm đè lên trên
        avg_data = final_summary_list[0] # Giả sử chỉ có 1 list tương ứng, hoặc filter lại
        # Tìm đúng summary của algo này
        subset_avg = all_summary[all_summary['run_label'] == algo_name]
        ax_d1.plot(subset_avg['round'], subset_avg['accuracy'], color='black', linewidth=2, linestyle='--', label='Average')
        ax_d2.plot(subset_avg['round'], subset_avg['loss'], color='black', linewidth=2, linestyle='--', label='Average')

        ax_d1.set_title(f'Node Accuracy Dispersion ({algo_name})')
        ax_d1.set_xlabel('Round'); ax_d1.set_ylabel('Accuracy')
        ax_d1.grid(True, alpha=0.3)
        ax_d1.legend(loc='lower right')

        ax_d2.set_title(f'Node Loss Dispersion ({algo_name})')
        ax_d2.set_xlabel('Round'); ax_d2.set_ylabel('Loss')
        ax_d2.grid(True, alpha=0.3)
        ax_d2.legend(loc='upper right')

        plt.suptitle(f'Detailed Performance of {len(nodes)} Nodes - {algo_name}', fontsize=16)
        plt.tight_layout()
        
        safe_name = algo_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f'chart_detail_{safe_name}.png')
        plt.show()

if __name__ == "__main__":
    main()