import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def get_latest_experiment(name_pattern):
    exp_dirs = sorted(glob(f"experiments/{name_pattern}*"))
    return exp_dirs[-1] if exp_dirs else None

def load_metrics(exp_dir):
    if not exp_dir: return None
    all_metrics = []
    jsonl_files = glob(os.path.join(exp_dir, "*.jsonl"))
    
    for f in jsonl_files:
        with open(f, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if "round" in data:
                        # Extract any key that looks like accuracy or loss
                        metrics = {"round": data["round"]}
                        for k, v in data.items():
                            if any(s in k.lower() for s in ["acc", "loss", "accuracy"]):
                                metrics[k] = v
                        all_metrics.append(metrics)
                except:
                    continue
    
    df = pd.DataFrame(all_metrics)
    if df.empty: return None
    return df.groupby("round").mean().reset_index()

# 1. Tìm thư mục theo tên experiment đã đặt trong YAML
# Cải tiến: Tìm theo prefix của 'name' trong experiment config
dir_v2 = get_latest_experiment("mnist_dirichlet_0.1_test_dfedadp2")
dir_orig = get_latest_experiment("mnist_dirichlet_0.1_test_dfedadp_original")

if not dir_v2 or not dir_orig:
    print("Trying alternative naming patterns...")
    dir_v2 = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_2")
    dir_orig = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_")

if not dir_v2 or not dir_orig:
    print(f"FAILED: Could not find experiment directories.\nV2: {dir_v2}\nOriginal: {dir_orig}")
    print("Available folders in experiments/:", os.listdir("experiments")[:5])
    exit()

print(f"Comparing:\n- V2: {dir_v2}\n- Original: {dir_orig}")

# 2. Load data
df_v2 = load_metrics(dir_v2)
df_orig = load_metrics(dir_orig)

if df_v2 is None or df_orig is None:
    print("Error: One of the dataframes is empty. Check if .jsonl files exist and contain metrics.")
    exit()

# Identify metric columns (excluding 'round')
metric_cols = [c for c in df_v2.columns if c != 'round']
acc_col = next((c for c in metric_cols if "acc" in c.lower()), None)
loss_col = next((c for c in metric_cols if "loss" in c.lower()), None)

print(f"Found metrics: {metric_cols}")

# 3. Plot
plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
if acc_col:
    if df_v2 is not None and acc_col in df_v2:
        plt.plot(df_v2["round"], df_v2[acc_col], label=f"V2 ({acc_col})", marker='o', linewidth=2)
    if df_orig is not None and acc_col in df_orig:
        plt.plot(df_orig["round"], df_orig[acc_col], label=f"Original ({acc_col})", marker='x', linestyle='--')
    plt.title("Accuracy Comparison")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
plt.grid(True, alpha=0.3)

# Loss Plot
plt.subplot(1, 2, 2)
if loss_col:
    if df_v2 is not None and loss_col in df_v2:
        plt.plot(df_v2["round"], df_v2[loss_col], label=f"V2 ({loss_col})", marker='o', linewidth=2)
    if df_orig is not None and loss_col in df_orig:
        plt.plot(df_orig["round"], df_orig[loss_col], label=f"Original ({loss_col})", marker='x', linestyle='--')
    plt.title("Loss Comparison")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_results.png", dpi=300)
print(f"\nSUCCESS: Plot saved to 'comparison_results.png'")
plt.show()
