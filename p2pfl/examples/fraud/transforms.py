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
from math import radians, sin, cos, sqrt, atan2

# Numeric feature columns for standardization
NUMERIC_FEATURES = [
    "amt",                    # Transaction amount
    "lat", "long",            # Customer location
    "city_pop",               # City population
    "merch_lat", "merch_long", # Merchant location
    "unix_time"               # Timestamp
]

# Feature stats for standardization (computed from training data)
FEATURE_STATS = {}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula."""
    R = 6371  # Earth radius in km
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    except (ValueError, TypeError):
        return 0.0


def compute_feature_statistics(dataset):
    """
    Compute mean and std for numeric features from dataset.
    
    Call this once on training data before transforming.
    """
    global FEATURE_STATS
    
    stats = {}
    for feat in NUMERIC_FEATURES:
        try:
            values = [float(x) for x in dataset[feat] if x is not None]
            if values:
                stats[feat] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
            else:
                stats[feat] = {"mean": 0.0, "std": 1.0}
        except (ValueError, TypeError):
            stats[feat] = {"mean": 0.0, "std": 1.0}
    
    FEATURE_STATS = stats
    print(f"Feature statistics computed for {len(stats)} features")


def fraud_transform(examples):
    """
    Transform batch of raw CSV rows into PyTorch tensors.
    
    Extracts numeric features and standardizes them.
    
    Args:
        examples: Dictionary with batch of CSV rows (each key is a list).
        
    Returns:
        Dictionary with 'features' (list of FloatTensors) and 'label' (list of LongTensors).
    """
    batch_size = len(examples.get("amt", []))
    features_list = []
    labels_list = []
    
    for idx in range(batch_size):
        # Extract numeric features in consistent order
        feature_values = []
        
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
