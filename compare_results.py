import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def get_latest_experiment(name_pattern):
    # Tìm thư mục experiment mới nhất khớp với pattern
    exp_dirs = sorted(glob(f"experiments/{name_pattern}*"))
    return exp_dirs[-1] if exp_dirs else None

def load_metrics(exp_dir):
    if not exp_dir: return None
    all_metrics = []
    # Tìm tất cả file .jsonl của các node
    jsonl_files = glob(os.path.join(exp_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {exp_dir}")
        return None

    for f in jsonl_files:
        with open(f, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if "round" in data:
                        entry = {"round": data["round"]}
                        # P2PFL lồng metrics vào trong key 'metrics'
                        m_data = data.get("metrics", {})
                        for k, v in m_data.items():
                            if any(s in k.lower() for s in ["acc", "loss", "accuracy", "f1"]):
                                entry[k] = v
                        all_metrics.append(entry)
                except Exception as e:
                    continue
    
    df = pd.DataFrame(all_metrics)
    if df.empty: return None
    # Tính trung bình cộng của tất cả các node tại mỗi round
    return df.groupby("round").mean().reset_index()

# 1. Xác định thư mục dựa trên kết quả bạn vừa pull về
dir_v2 = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_2")
dir_orig = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_model") 
# Lưu ý: dir_orig sẽ lấy folder không có số '2', tức là bản gốc 20260418_073713

if not dir_v2 or not dir_orig:
    print(f"FAILED: Could not find experiment directories.")
    print(f"V2 Pattern: p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_2* -> Found: {dir_v2}")
    print(f"Orig Pattern: p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_model* -> Found: {dir_orig}")
    exit()

print(f"Comparing:\n- Improved (V2): {dir_v2}\n- Original: {dir_orig}")

# 2. Load data
df_v2 = load_metrics(dir_v2)
df_orig = load_metrics(dir_orig)

if df_v2 is None or df_orig is None:
    print("Error: Could not load metrics from one of the directories.")
    exit()

# 3. Plotting
plt.style.use('seaborn-v0_8') # Thêm style cho đẹp
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy Plot
acc_key = "test_acc"
if acc_key in df_v2.columns:
    ax1.plot(df_v2["round"], df_v2[acc_key], 'o-', label="DFedAdp_2 (Improved)", linewidth=2)
    ax1.plot(df_orig["round"], df_orig[acc_key], 's--', label="DFedAdp (Original)", alpha=0.8)
    ax1.set_title("Test Accuracy Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

# Loss Plot
loss_key = "test_loss"
if loss_key in df_v2.columns:
    ax2.plot(df_v2["round"], df_v2[loss_key], 'o-', color='red', label="DFedAdp_2 (Improved)", linewidth=2)
    ax2.plot(df_orig["round"], df_orig[loss_key], 's--', color='orange', label="DFedAdp (Original)", alpha=0.8)
    ax2.set_title("Test Loss Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

plt.tight_layout()
plt.savefig("comparison_results.png", dpi=300)
print("\nSUCCESS! Biểu đồ so sánh đã được lưu tại 'comparison_results.png'")
plt.show()
