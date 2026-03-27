import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from collections import defaultdict

# --- Configuration ---
EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_nodes_metrics(exp_dir):
    """Calculate mean metrics across all nodes per round."""
    jsonl_files = glob.glob(os.path.join(exp_dir, "node_node*.jsonl"))
    if not jsonl_files:
        return None, None

    round_accs = defaultdict(list)
    
    for f_path in jsonl_files:
        try:
            with open(f_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        r = data['round']
                        m = data['metrics']
                        round_accs[r].append(m.get('test_acc', 0))
        except Exception:
            continue
            
    rounds = sorted(round_accs.keys())
    if not rounds: return None, None

    mean_accs = [np.mean(round_accs[r]) for r in rounds]
    return rounds, mean_accs

def plot_ablation():
    # Find ablation experiment folders
    # They should start with 'ablation_mnist_'
    folders = [f for f in os.listdir(EXPERIMENTS_DIR) if f.startswith("ablation_mnist_")]
    # Group by base name (without timestamp)
    groups = defaultdict(list)
    for f in folders:
        # Expected format: ablation_mnist_full_20260327_123456
        parts = f.split('_')
        # Join everything except the last two (date and time)
        base_name = "_".join(parts[:-2])
        groups[base_name].append(f)
    
    # For each group, pick the latest one
    latest_folders = {}
    for base_name, folder_list in groups.items():
        folder_list.sort(reverse=True)
        latest_folders[base_name] = os.path.join(EXPERIMENTS_DIR, folder_list[0])
    
    if not latest_folders:
        print("No ablation experiment results found in 'experiments/' folder.")
        return

    plt.figure(figsize=(12, 8))
    
    # Sort for consistent legend
    for label in sorted(latest_folders.keys()):
        folder = latest_folders[label]
        print(f"Loading {label} from {folder}...")
        rounds, mean_accs = load_all_nodes_metrics(folder)
        if rounds:
            # Clean up label for display
            display_label = label.replace("ablation_mnist_", "").replace("_", " ").title()
            plt.plot(rounds, mean_accs, label=display_label, linewidth=2.5)

    plt.title("DFedAdp Ablation Study: MNIST Test Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=13)
    plt.ylabel("Global Mean Test Accuracy", fontsize=13)
    plt.ylim([0, 1.0])
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "Ablation_Comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Ablation comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_ablation()
