import json
import matplotlib.pyplot as plt
import os

def plot_jsonl_metrics(file_path):
    rounds = []
    losses = []
    accuracies = []
    f1_scores = []

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    rounds.append(data['round'])
                    m = data['metrics']
                    losses.append(m.get('test_loss', 0))
                    accuracies.append(m.get('test_acc', 0))
                    f1_scores.append(m.get('test_f1', 0))
                except Exception as e:
                    print(f"Skip bad line: {e}")

    if not rounds:
        print("No data found to plot.")
        return

    # 2. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Accuracy & F1 (RAW DATA ONLY)
    ax1.plot(rounds, accuracies, color='#1f77b4', marker='o', markersize=3, linewidth=1.5, label='Test Accuracy')
    ax1.plot(rounds, f1_scores, color='#2ca02c', marker='x', markersize=3, linewidth=1, linestyle=':', label='Test F1 Score')
    
    ax1.set_ylabel('Score')
    ax1.set_title(f"Node Performance Metrics (Raw)\nFile: {os.path.basename(file_path)}")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust Y-axis for accuracy
    if accuracies:
        ax1.set_ylim([min(accuracies)*0.98, 1.01])

    # Plot 2: Loss (RAW DATA ONLY)
    ax2.plot(rounds, losses, color='#d62728', marker='.', markersize=3, linewidth=1.5, label='Test Loss')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save & Show
    output_img = "raw_node_metrics.png"
    plt.savefig(output_img, dpi=300)
    print(f"Successfully saved raw plot to: {output_img}")
    plt.show()

if __name__ == "__main__":
    # This node has the best convergence (Loss 0.59 -> 0.08)
    target_file = os.path.join(
        "experiments", 
        "p2pfl_MNIST_DirichletPartitionStrategy_DFedAdp_model_build_fn_20260215_134046", 
        "node_0.0.0.0_33141.jsonl"
    )
    plot_jsonl_metrics(target_file)
