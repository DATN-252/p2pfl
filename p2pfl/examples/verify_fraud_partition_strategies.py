"""
A standalone script to verify and visualize data partitioning strategies for the Fraud Detection dataset.

This script loads the Kaggle 'kartik2112/fraud-detection' dataset, applies different 
partitioning strategies (Random IID, Dirichlet), and generates plots to visualize 
the class distribution across the resulting partitions.

Note: This script requires the 'kaggle' library and a configured Kaggle API key.
See: https://www.kaggle.com/docs/api

To run:
    python -m p2pfl.examples.verify_fraud_partition_strategies
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import (
    DirichletPartitionStrategy,
    RandomIIDPartitionStrategy,
    LabelSkewedPartitionStrategy
)

LABEL_COLUMN = "is_fraud"
NUM_CLASSES = 2


def analyze_partitions(partitions: list[P2PFLDataset]) -> pd.DataFrame:
    """
    Analyzes the class distribution of a list of P2PFLDataset partitions.

    Args:
        partitions: A list of P2PFLDataset objects, each representing a partition.

    Returns:
        A pandas DataFrame where rows are partition indices, columns are class labels,
        and values are the number of samples for each class in that partition.
    """
    partition_analysis = pd.DataFrame(np.zeros((len(partitions), NUM_CLASSES)))

    for i, partition in enumerate(partitions):
        # Ensure the partition has a 'train' split and it's not empty
        if "train" not in partition._data or len(partition._data["train"]) == 0:
            continue

        df = partition._data["train"].to_pandas()
        if LABEL_COLUMN not in df.columns:
            print(f"Warning: Partition {i} has no '{LABEL_COLUMN}' column.")
            continue
            
        counts = df[LABEL_COLUMN].value_counts().sort_index()
        partition_analysis.loc[i, counts.index] = counts.values

    partition_analysis.columns = [f"Class {i}" for i in range(NUM_CLASSES)]
    partition_analysis.index.name = "Partition ID"
    return partition_analysis


def plot_distribution(
    distribution_df: pd.DataFrame, title: str, output_path: str
):
    """
    Plots the class distribution as a heatmap.

    Args:
        distribution_df: DataFrame containing the class distribution data.
        title: The title for the plot.
        output_path: Path to save the plot image.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        distribution_df,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Number of Samples"},
    )
    plt.title(title, fontsize=16)
    plt.ylabel("Partition (Node) ID")
    plt.xlabel("Class Label")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    """Main function to run the verification."""
    # --- Configuration ---
    NUM_PARTITIONS = 4
    # Use a subset for faster processing
    DATASET_SIZE = 10000
    DIRICHLET_ALPHAS = [0.1, 0.5, 10]  # Low, medium, and high alpha values
    SHARDS_PER_NODE = 2

    print("Loading Fraud Detection dataset from Kaggle...")
    try:
        # Load the full dataset from Kaggle
        kaggle_dataset = load_dataset("kaggle/kartik2112/fraud-detection")
        
        # The Kaggle loader might name splits based on filenames, e.g., 'fraudTrain', 'fraudTest'
        # We need to map them to 'train' and 'test'
        train_split_name = 'fraudTrain'
        test_split_name = 'fraudTest'

        if train_split_name not in kaggle_dataset or test_split_name not in kaggle_dataset:
            raise ValueError(f"Could not find '{train_split_name}' or '{test_split_name}' splits.")

        # Take a subset for faster processing
        train_subset = kaggle_dataset[train_split_name].select(range(DATASET_SIZE))
        test_subset = kaggle_dataset[test_split_name].select(range(1000))

        dataset = P2PFLDataset(
            DatasetDict(
                {
                    "train": train_subset,
                    "test": test_subset,
                }
            )
        )
        print(f"Dataset loaded with {dataset.get_num_samples(train=True)} training samples.")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have the 'kaggle' library installed (`pip install kaggle`)")
        print("and your Kaggle API credentials are correctly configured in ~/.kaggle/kaggle.json")
        return

    # --- 1. Verify RandomIIDPartitionStrategy ---
    print("\nVerifying RandomIIDPartitionStrategy...")
    iid_partitions = dataset.generate_partitions(
        num_partitions=NUM_PARTITIONS, 
        strategy=RandomIIDPartitionStrategy,
        label_column=LABEL_COLUMN
    )
    iid_distribution = analyze_partitions(iid_partitions)
    plot_distribution(
        iid_distribution,
        f"IID Partition Distribution ({NUM_PARTITIONS} Partitions) on Fraud Dataset",
        "fraud_iid_distribution.png",
    )

    # --- 2. Verify DirichletPartitionStrategy ---
    for alpha in DIRICHLET_ALPHAS:
        print(f"\nVerifying DirichletPartitionStrategy with alpha={alpha}...")
        dirichlet_partitions = dataset.generate_partitions(
            num_partitions=NUM_PARTITIONS,
            strategy=DirichletPartitionStrategy,
            alpha=alpha,
            label_column=LABEL_COLUMN
        )
        dirichlet_distribution = analyze_partitions(dirichlet_partitions)
        plot_distribution(
            dirichlet_distribution,
            f"Dirichlet Partition (alpha={alpha}, {NUM_PARTITIONS} Partitions) on Fraud Dataset",
            f"fraud_dirichlet_distribution_alpha_{alpha}.png",
        )
        
    # --- 3. Verify LabelSkewedPartitionStrategy ---
    print(f"\nVerifying LabelSkewedPartitionStrategy with {SHARDS_PER_NODE} shards per node...")
    label_skewed_partitions = dataset.generate_partitions(
        num_partitions=NUM_PARTITIONS,
        strategy=LabelSkewedPartitionStrategy,
        shards_per_node=SHARDS_PER_NODE,
        label_column=LABEL_COLUMN
    )
    label_skewed_distribution = analyze_partitions(label_skewed_partitions)
    plot_distribution(
        label_skewed_distribution,
        f"Label Skewed Partition ({SHARDS_PER_NODE} Shards/Node, {NUM_PARTITIONS} Partitions) on Fraud Dataset",
        "fraud_label_skewed_distribution.png",
    )

    print("\nVerification complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
