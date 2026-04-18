import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def get_latest_experiment(name_pattern):
    exp_dirs = sorted(glob(f"experiments/{name_pattern}*"))
    return exp_dirs[-1] if exp_dirs else None

def load_metrics(exp_dir):
    all_metrics = []
    # Find all node jsonl files in the experiment directory
    jsonl_files = glob(os.path.join(exp_dir, "node_*.jsonl"))
    
    for f in jsonl_files:
        with open(f, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if "round" in data and "test_acc" in data:
                        all_metrics.append({
                            "round": data["round"],
                            "test_acc": data["test_acc"],
                            "test_loss": data.get("test_loss", 0)
                        })
                except:
                    continue
    
    df = pd.DataFrame(all_metrics)
    if df.empty:
        return None
    # Group by round and take average across nodes
    return df.groupby("round").mean().reset_index()

# 1. Get directories
dir_v2 = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_2")
dir_orig = get_latest_experiment("p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_")

if not dir_v2 or not dir_orig:
    print(f"Could not find experiment directories.\nV2: {dir_v2}\nOriginal: {dir_orig}")
    print("Please run both experiments first.")
    exit()

print(f"Comparing:\n- V2: {dir_v2}\n- Original: {dir_orig}")

# 2. Load data
df_v2 = load_metrics(dir_v2)
df_orig = load_metrics(dir_orig)

# 3. Plot
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
if df_v2 is not None:
    plt.plot(df_v2["round"], df_v2["test_acc"], label="DFedAdp_2 (Improved)", marker='o')
if df_orig is not None:
    plt.plot(df_orig["round"], df_orig["test_acc"], label="DFedAdp (Original)", marker='x')
plt.title("Test Accuracy Comparison")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
if df_v2 is not None:
    plt.plot(df_v2["round"], df_v2["test_loss"], label="DFedAdp_2 (Improved)", marker='o')
if df_orig is not None:
    plt.plot(df_orig["round"], df_orig["test_loss"], label="DFedAdp (Original)", marker='x')
plt.title("Test Loss Comparison")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("comparison_results.png")
print("\nSuccess! Comparison plot saved as 'comparison_results.png'")
plt.show()
