#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#

"""Transform functions for fraud detection dataset."""

import torch
import pandas as pd
import numpy as np
from typing import Any

# Numeric feature columns for standardization
NUMERIC_FEATURES = [
    "amt",
    "distance_km",
    "hour",
    "age",
    "merchant_freq",
    "dayofweek"
]

# Categorical feature columns (one-hot encoded)
CATEGORICAL_FEATURES = [
    "cat_entertainment",
    "cat_food_dining",
    "cat_gas_transport",
    "cat_grocery_net",
    "cat_grocery_pos",
    "cat_health_fitness",
    "cat_home",
    "cat_kids_pets",
    "cat_misc_net",
    "cat_misc_pos",
    "cat_personal_care",
    "cat_shopping_net",
    "cat_shopping_pos",
    "cat_travel"
]

# All features in order
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Feature stats for standardization (computed from training data)
FEATURE_STATS = {}


def compute_feature_statistics(dataset):
    """
    Compute mean and std for numeric features from dataset.
    
    Call this once on training data before transforming.
    
    Args:
        dataset: A list of dictionaries or a Hugging Face Dataset containing the training data.
    """
    global FEATURE_STATS
    
    # Convert Hugging Face Dataset to DataFrame if needed
    if hasattr(dataset, 'to_pandas'):
        df = dataset.to_pandas()
    elif isinstance(dataset, list):
        df = pd.DataFrame(dataset)
    else:
        df = dataset
    
    # Compute statistics for numeric features
    stats = {}
    for feat in NUMERIC_FEATURES:
        try:
            if feat in df.columns:
                values = [float(x) for x in df[feat] if x is not None and not pd.isna(x)]
                if values:
                    stats[feat] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }
                else:
                    stats[feat] = {"mean": 0.0, "std": 1.0}
            else:
                stats[feat] = {"mean": 0.0, "std": 1.0}
        except (ValueError, TypeError):
            stats[feat] = {"mean": 0.0, "std": 1.0}
    
    FEATURE_STATS = stats
    print(f"Feature statistics computed for {len(stats)} numeric features")
    print(f"Total features (including categorical): {len(ALL_FEATURES)}")


def fraud_transform(examples):
    """
    Transform batch of raw CSV rows into PyTorch tensors.
    
    Extracts specified features and standardizes them.
    
    Args:
        examples: Dictionary with batch of CSV rows (each key is a list).
        
    Returns:
        Dictionary with 'features' (list of FloatTensors) and 'label' (list of LongTensors).
    """
    batch_size = len(examples.get("amt", []))
    features_list = []
    labels_list = []
    
    for idx in range(batch_size):
        feature_values = []
        
        # Process numeric features
        for feat in NUMERIC_FEATURES:
            if feat in examples and idx < len(examples[feat]):
                try:
                    val = float(examples[feat][idx] or 0)
                except (ValueError, TypeError):
                    val = 0.0
            else:
                val = 0.0
            
            # Standardize if stats available
            if feat in FEATURE_STATS and FEATURE_STATS[feat]["std"] > 0:
                stats = FEATURE_STATS[feat]
                val = (val - stats["mean"]) / stats["std"]
            
            feature_values.append(val)
        
        # Process categorical features (one-hot encoded)
        for feat in CATEGORICAL_FEATURES:
            if feat in examples and idx < len(examples[feat]):
                try:
                    val = float(examples[feat][idx] or 0)
                except (ValueError, TypeError):
                    val = 0.0
            else:
                val = 0.0
            feature_values.append(val)
        
        # Create feature tensor
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)
        features_list.append(feature_tensor)
        
        # Create label tensor
        try:
            is_fraud = int(examples.get("is_fraud", [0])[idx] or 0)
        except (ValueError, TypeError, IndexError):
            is_fraud = 0
        
        label_tensor = torch.tensor(is_fraud, dtype=torch.long)
        labels_list.append(label_tensor)
    
    return {
        "features": features_list,
        "label": labels_list
    }


def get_fraud_transforms():
    """Export fraud transforms (unified for train/test)."""
    return {"train": fraud_transform, "test": fraud_transform}


def setup_fraud_transforms(dataset: Any) -> dict[str, Any]:
    """
    Setup fraud transforms by computing feature statistics on training data.
    
    This function should be called ONCE on the full training dataset BEFORE
    applying transforms to individual partitions. It computes statistics and
    returns the transforms dictionary.
    
    Args:
        dataset: The full training dataset (Hugging Face Dataset, list of dicts, or DataFrame).
    
    Returns:
        Dictionary with 'train' and 'test' transform functions.
        
    Example:
        >>> from p2pfl.examples.fraud.transforms import setup_fraud_transforms
        >>> from datasets import load_dataset
        >>> 
        >>> # Load dataset
        >>> dataset = load_dataset("csv", data_files="fraudTrain.csv")
        >>> 
        >>> # Setup transforms (computes statistics on full training data)
        >>> transforms = setup_fraud_transforms(dataset["train"])
        >>> 
        >>> # Apply transforms to partitions
        >>> for partition in partitions:
        >>>     partition.set_transforms(transforms)
    """
    # Compute statistics on the full training dataset
    compute_feature_statistics(dataset)
    
    # Return the transforms dictionary
    return {"train": fraud_transform, "test": fraud_transform}


def get_num_features() -> int:
    """
    Get the number of features after transformation.
    
    Returns:
        Total number of features.
    """
    return len(ALL_FEATURES)
