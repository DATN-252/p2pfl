import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_results(exp_dir):
    jsonl_files = glob.glob(os.path.join(exp_dir, "node_node*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {exp_dir}")
        return

    all_data = []
    for f in jsonl_files:
        node_data = []
        with open(f, 'r') as file:
            for line in file:
                data = json.loads(line)
                row = {'round': data['round']}
                row.update(data['metrics'])
                node_data.append(row)
        all_data.append(pd.DataFrame(node_data))

    # Combine all nodes and calculate average per round
    df_all = pd.concat(all_data)
    df_avg = df_all.groupby('round').mean().reset_index()

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Average Metrics for Experiment:\n{os.path.basename(exp_dir)}", fontsize=16)

    # 1. Accuracy
    axes[0, 0].plot(df_avg['round'], df_avg['test_accuracy'], label='Accuracy', color='blue')
    axes[0, 0].set_title('Test Accuracy')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)

    # 2. Loss
    axes[0, 1].plot(df_avg['round'], df_avg['test_loss'], label='Loss', color='red')
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)

    # 3. Precision & Recall
    axes[1, 0].plot(df_avg['round'], df_avg['test_precision'], label='Precision', color='green')
    axes[1, 0].plot(df_avg['round'], df_avg['test_recall'], label='Recall', color='orange')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. F1 Score
    axes[1, 1].plot(df_avg['round'], df_avg['test_f1'], label='F1 Score', color='purple')
    axes[1, 1].set_title('Test F1 Score')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_plot = os.path.join(exp_dir, "average_metrics_plot.png")
    plt.savefig(output_plot)
    print(f"Plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    exp_path = "experiments/p2pfl_examples_fraud_processed_data_train_processed.csv_RandomIIDPartitionStrategy_DFedAdp_model_build_fn_20260509_022641"
    plot_results(exp_path)
