"""
A standalone script to verify and visualize data partitioning strategies.

This script loads the MNIST dataset, applies different partitioning strategies 
(Random IID, Dirichlet, and Label Skewed), and generates plots to visualize 
the class distribution across the resulting partitions.

To run:
    python -m p2pfl.examples.verify_partition_strategies
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DatasetDict, load_dataset

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import (
    DirichletPartitionStrategy,
    LabelSkewedPartitionStrategy,
    QualitySkewedPartitionStrategy,
    RandomIIDPartitionStrategy,
)


def analyze_partitions(partitions: list[P2PFLDataset]) -> pd.DataFrame:
    """
    Analyzes the class distribution of a list of P2PFLDataset partitions.

    Args:
        partitions: A list of P2PFLDataset objects, each representing a partition.

    Returns:
        A pandas DataFrame where rows are partition indices, columns are class labels,
        and values are the number of samples for each class in that partition.
    """
    num_classes = 10  # For MNIST

    column_names = [f"Class {i}" for i in range(num_classes)] + ["Total"]
    partition_analysis = pd.DataFrame(
        0, 
        index=[f"Partition {i}" for i in range(len(partitions))] + ["Total Sum"], 
        columns=column_names
    )
    partition_analysis.index.name = "Partition ID"

    for i, partition in enumerate(partitions):
        # Ensure the partition has a 'train' split and it's not empty
        if "train" not in partition._data or len(partition._data["train"]) == 0:
            continue

        df = partition._data["train"].to_pandas()
        if "label" not in df.columns:
            print(f"Warning: Partition {i} has no 'label' column.")
            continue
        
        counts = df["label"].value_counts().sort_index()
        for label, count in counts.items():
            partition_analysis.loc[f"Partition {i}", f"Class {label}"] = count
        partition_analysis.loc[f"Partition {i}", "Total"] = counts.sum()

    partition_analysis.loc["Total Sum"] = partition_analysis.sum(axis=0)
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
    NUM_PARTITIONS = 10
    # Use a subset for faster processing
    DATASET_SIZE = 5000
    DIRICHLET_ALPHAS = [0.1, 0.5, 10]  # Low, medium, and high alpha values
    QUALITY_ALPHAS = [0.5, 1, 5, 10]  # Low, medium, and high alpha values
    LABLES_ALPHAS = [0.5, 1, 5, 10]  # Low, medium, and high alpha values
    SHARDS_PER_NODE = 4

    print("Loading MNIST dataset...")
    # Load a subset of the MNIST dataset
    dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split=f"train[:{DATASET_SIZE}]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:100]"),
            }
        )
    )
    print(f"Dataset loaded with {dataset.get_num_samples(train=True)} training samples.")

    # --- 1. Verify RandomIIDPartitionStrategy ---
    print("\nVerifying RandomIIDPartitionStrategy...")
    iid_partitions = dataset.generate_partitions(
        num_partitions=NUM_PARTITIONS, strategy=RandomIIDPartitionStrategy
    )
    iid_distribution = analyze_partitions(iid_partitions)
    plot_distribution(
        iid_distribution,
        f"IID Partition Distribution ({NUM_PARTITIONS} Partitions)",
        "iid_distribution.png",
    )

    # # --- 2. Verify DirichletPartitionStrategy ---
    # for alpha in DIRICHLET_ALPHAS:
    #     print(f"\nVerifying DirichletPartitionStrategy with alpha={alpha}...")
    #     dirichlet_partitions = dataset.generate_partitions(
    #         num_partitions=NUM_PARTITIONS,
    #         strategy=DirichletPartitionStrategy,
    #         alpha=alpha,
    #     )
    #     dirichlet_distribution = analyze_partitions(dirichlet_partitions)
    #     plot_distribution(
    #         dirichlet_distribution,
    #         f"Dirichlet Partition (alpha={alpha}, {NUM_PARTITIONS} Partitions)",
    #         f"dirichlet_distribution_alpha_{alpha}.png",
    #     )
        
    # # --- 3. Verify LabelSkewedPartitionStrategy ---
    # print(f"\nVerifying LabelSkewedPartitionStrategy with {SHARDS_PER_NODE} shards per node...")
    # label_skewed_partitions = dataset.generate_partitions(
    #     num_partitions=NUM_PARTITIONS,
    #     strategy=LabelSkewedPartitionStrategy,
    #     shards_per_node=SHARDS_PER_NODE,
    # )
    # label_skewed_distribution = analyze_partitions(label_skewed_partitions)
    # plot_distribution(
    #     label_skewed_distribution,
    #     f"Label Skewed Partition ({SHARDS_PER_NODE} Shards/Node, {NUM_PARTITIONS} Partitions)",
    #     "label_skewed_distribution.png",
    # )
    
    # --- 4. Verify QualitySkewedPartitionStrategy ---
    for q_a in QUALITY_ALPHAS:
        for l_a in LABLES_ALPHAS:
            print(f"\nVerifying QualitySkewedPartitionStrategy with quanlity alpha={q_a} and label alpha={l_a}...")
            quality_partitions = dataset.generate_partitions(
                num_partitions=NUM_PARTITIONS,
                strategy=QualitySkewedPartitionStrategy, # type: ignore
                alpha_quantity=q_a,     # Quantity Skew
                alpha_label=l_a         # Label Skew
            )
            quality_distribution = analyze_partitions(quality_partitions)

            if not os.path.exists("quality_distribution"):
                os.makedirs("quality_distribution") 

            plot_distribution(
                quality_distribution,
                f"Quality-Skewed Partition (alpha={q_a} and label alpha={l_a}, {NUM_PARTITIONS} Partitions)",
                f"quality_distribution/quality_distribution_qa_{q_a}_la_{l_a}.png",
            )
    
    
    print("\nVerification complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
