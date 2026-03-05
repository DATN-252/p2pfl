import json
import matplotlib.pyplot as plt
import os
import yaml
import glob
import numpy as np
from collections import defaultdict

# --- Configuration ---
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_experiment_details(exp_dir):
    """Extract partition info and alpha directly from yaml."""
    yaml_files = glob.glob(os.path.join(exp_dir, "*.yaml"))
    if not yaml_files:
        return {"label": os.path.basename(exp_dir), "sort_key": 999, "file_name": "Unknown"}
    
    with open(yaml_files[0], 'r') as f:
        config = yaml.safe_load(f)
        strategy = config.get("experiment", {}).get("dataset", {}).get("partitioning", {}).get("strategy", "Unknown")
        params = config.get("experiment", {}).get("dataset", {}).get("partitioning", {}).get("params", {})
        
        if "Dirichlet" in strategy:
            alpha = params.get('alpha', 0)
            return {
                "label": f"Dirichlet (α={alpha})",
                "sort_key": 100 - alpha,
                "file_name": f"Dirichlet_alpha_{alpha}"
            }
        elif "LabelSkewed" in strategy:
            shards = params.get('shards_per_node', '?')
            return {
                "label": f"Label Skewed ({shards} shards)",
                "sort_key": 500,
                "file_name": f"LabelSkewed_{shards}shards"
            }
        elif "RandomIID" in strategy:
            return {
                "label": "Random IID",
                "sort_key": 0,
                "file_name": "Random_IID"
            }
        return {"label": strategy, "sort_key": 900, "file_name": strategy}

def load_all_nodes_metrics(exp_dir):
    """Calculate mean and std metrics across all nodes per round."""
    jsonl_files = glob.glob(os.path.join(exp_dir, "node_node*.jsonl"))
    if not jsonl_files:
        return None, None, None, None

    round_accs = defaultdict(list)
    round_losses = defaultdict(list)
    
    for f_path in jsonl_files:
        try:
            with open(f_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        r = data['round']
                        m = data['metrics']
                        round_accs[r].append(m.get('test_acc', 0))
                        round_losses[r].append(m.get('test_loss', 0))
        except Exception:
            continue
            
    rounds = sorted(round_accs.keys())
    if not rounds: return None, None, None, None

    mean_accs = [np.mean(round_accs[r]) for r in rounds]
    std_accs = [np.std(round_accs[r]) for r in rounds]
    mean_losses = [np.mean(round_losses[r]) for r in rounds]
    std_losses = [np.std(round_losses[r]) for r in rounds]
    
    return rounds, mean_accs, std_accs, mean_losses, std_losses

def save_single_exp_plot(data):
    """Save a detailed plot for a single experiment."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    r = data['rounds']
    
    # Accuracy Plot
    ax1.plot(r, data['mean_accs'], color='tab:blue', linewidth=2, label='Mean Accuracy')
    ax1.fill_between(r, 
                     np.array(data['mean_accs']) - np.array(data['std_accs']), 
                     np.array(data['mean_accs']) + np.array(data['std_accs']), 
                     color='tab:blue', alpha=0.2, label='Std Dev')
    ax1.set_title(f"Experiment Performance: {data['label']}", fontsize=14)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Loss Plot
    ax2.plot(r, data['mean_losses'], color='tab:red', linewidth=2, label='Mean Loss')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f"Exp_{data['file_name']}.png")
    plt.savefig(file_path, dpi=200)
    plt.close()
    print(f"   -> Saved individual plot: {file_path}")

def batch_plot(exp_dirs):
    plot_data = []
    for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            print(f"Warning: {exp_dir} not found.")
            continue
            
        details = get_experiment_details(exp_dir)
        results = load_all_nodes_metrics(exp_dir)
        
        if results[0]:
            data = {
                "label": details['label'],
                "file_name": details['file_name'],
                "sort_key": details['sort_key'],
                "rounds": results[0],
                "mean_accs": results[1],
                "std_accs": results[2],
                "mean_losses": results[3]
            }
            plot_data.append(data)
            # Save individual plot
            save_single_exp_plot(data)
    
    # Sort for summary plot
    plot_data.sort(key=lambda x: x['sort_key'])

    # --- Summary Comparison Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(plot_data)))
    
    print("\nGenerating Summary Comparison...")
    for i, data in enumerate(plot_data):
        color = colors[i]
        ax1.plot(data['rounds'], data['mean_accs'], label=data['label'], color=color, linewidth=2.5)
        ax1.fill_between(data['rounds'], 
                         np.array(data['mean_accs']) - np.array(data['std_accs']), 
                         np.array(data['mean_accs']) + np.array(data['std_accs']), 
                         color=color, alpha=0.1)
        
        ax2.plot(data['rounds'], data['mean_losses'], label=data['label'], color=color, linewidth=2)

    ax1.set_title("DFedAdp: Global Mean Accuracy (Comparison)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Test Accuracy", fontsize=13)
    ax1.set_ylim([0, 1.0])
    ax1.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    ax2.set_title("Global Mean Convergence (Loss Comparison)", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Communication Round", fontsize=13)
    ax2.set_ylabel("Loss", fontsize=13)
    ax2.legend(loc='upper right', fontsize=10, ncol=2)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "Summary_Comparison.png")
    plt.savefig(summary_path, dpi=300)
    print(f"\n✅ All plots generated in folder: '{OUTPUT_DIR}'")
    ax1.set_title("Q-DFedAvgM: Global Mean Accuracy (Comparison)", fontsize=16, fontweight='bold')
 
if __name__ == "__main__":
    base_dir = "experiments"
    experiments = [
        os.path.join(base_dir, "p2pfl_MNIST_RandomIIDPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260304_155902"),
        os.path.join(base_dir, "p2pfl_MNIST_DirichletPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260304_201208"),
        os.path.join(base_dir, "p2pfl_MNIST_DirichletPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260304_223815"),
        os.path.join(base_dir, "p2pfl_MNIST_DirichletPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260305_005609"),
        os.path.join(base_dir, "p2pfl_MNIST_DirichletPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260305_024823"),
        os.path.join(base_dir, "p2pfl_MNIST_LabelSkewedPartitionStrategy_QDFedAvgMAggregator_model_build_fn_20260305_044635")
    ]
    batch_plot(experiments)

