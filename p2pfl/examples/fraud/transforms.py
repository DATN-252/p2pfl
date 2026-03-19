#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#

"""Transform functions for fraud detection dataset with online behavioral engineering (Phase 3)."""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Features to use (15 total)
NUMERIC_FEATURES = [
    "city_pop", "merch_lat", "merch_long", 
    "distance", "hour", "day_of_week", "category_idx", "age", "unix_time",
    "amt_diff_avg_30d", "trans_count_24h", "distance_velocity"
]

# Mapping for categories
CATEGORY_MAP = {
    'misc_net': 0, 'grocery_pos': 1, 'entertainment': 2, 'gas_transport': 3,
    'misc_pos': 4, 'grocery_net': 5, 'shopping_net': 6, 'shopping_pos': 7,
    'food_dining': 8, 'personal_care': 9, 'health_fitness': 10, 'travel': 11,
    'kids_pets': 12, 'home': 13
}

# Global lookups for behavioral features
FEATURE_STATS = {}
BEHAVIORAL_LOOKUP = {} # Key: (cc_num, unix_time), Value: {features}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates."""
    R = 6371  # km
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))
    except: return 0.0

def calculate_age(dob_str):
    """Calculate age from date of birth string."""
    try:
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        return float(datetime.now().year - dob.year)
    except: return 40.0

def build_behavioral_lookup(examples):
    """Pre-calculate advanced behavioral features including velocity."""
    global BEHAVIORAL_LOOKUP
    import gc
    
    df = pd.DataFrame(examples)
    # Ensure memory-heavy columns are converted efficiently
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed')
    df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])
    
    # 1. Rolling windows
    temp_df = df.set_index('trans_date_trans_time')
    df['avg_amt_30d'] = temp_df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window='30D', min_periods=1).mean()).values
    df['trans_count_24h'] = temp_df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(window='24h', min_periods=1).count()).values
    
    # 2. Distance Velocity (km/h)
    df['prev_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
    df['prev_long'] = df.groupby('cc_num')['merch_long'].shift(1)
    df['prev_time'] = df.groupby('cc_num')['unix_time'].shift(1)
    
    # Calculate distance to previous transaction
    lat1, lon1 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
    lat2, lon2 = np.radians(df['prev_lat'].fillna(df['merch_lat'])), np.radians(df['prev_long'].fillna(df['merch_long']))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist_to_prev = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    time_diff_h = (df['unix_time'] - df['prev_time']).fillna(3600) / 3600.0
    df['distance_velocity'] = (dist_to_prev / time_diff_h).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Fill lookup table
    for _, row in df.iterrows():
        key = (row['cc_num'], int(row['unix_time']))
        BEHAVIORAL_LOOKUP[key] = {
            'amt_diff_avg_30d': float(row['amt'] - row['avg_amt_30d']),
            'trans_count_24h': float(row['trans_count_24h']),
            'distance_velocity': float(row['distance_velocity'])
        }
    
    # Explicitly clear large objects and trigger GC
    del df
    del temp_df
    gc.collect()

def fraud_transform(examples):
    """Transform batch using 15 optimized features."""
    global FEATURE_STATS, BEHAVIORAL_LOOKUP
    batch_size = len(examples.get("amt", []))
    
    # Initialize behavioral lookup once per Node lifecycle
    if not BEHAVIORAL_LOOKUP and batch_size > 1:
        build_behavioral_lookup(examples)
    
    data_dict = {feat: [] for feat in NUMERIC_FEATURES}
    
    for idx in range(batch_size):
        # 1. Raw numeric
        for f in ["city_pop", "merch_lat", "merch_long", "unix_time"]:
            data_dict[f].append(float(examples.get(f, [0])[idx] or 0))
        
        # 2. Geospatial & Temporal
        data_dict["distance"].append(haversine(examples["lat"][idx], examples["long"][idx], 
                                             examples["merch_lat"][idx], examples["merch_long"][idx]))
        try:
            dt = datetime.strptime(examples["trans_date_trans_time"][idx], '%Y-%m-%d %H:%M:%S')
            data_dict["hour"].append(float(dt.hour))
            data_dict["day_of_week"].append(float(dt.weekday()))
        except:
            data_dict["hour"].append(0.0); data_dict["day_of_week"].append(0.0)
            
        data_dict["category_idx"].append(float(CATEGORY_MAP.get(examples.get("category", [""])[idx], 14)))
        data_dict["age"].append(calculate_age(examples.get("dob", ["1980-01-01"])[idx]))
        
        # 3. Behavioral Lookup
        key = (examples['cc_num'][idx], int(examples['unix_time'][idx]))
        beh = BEHAVIORAL_LOOKUP.get(key, {'amt_diff_avg_30d': 0.0, 'trans_count_24h': 1.0, 'distance_velocity': 0.0})
        data_dict['amt_diff_avg_30d'].append(beh['amt_diff_avg_30d'])
        data_dict['trans_count_24h'].append(beh['trans_count_24h'])
        data_dict['distance_velocity'].append(beh['distance_velocity'])

    df = pd.DataFrame(data_dict)
    
    # Auto-initialize stats if empty
    if not FEATURE_STATS:
        for feat in NUMERIC_FEATURES:
            m, s = df[feat].mean(), df[feat].std()
            FEATURE_STATS[feat] = {"mean": float(m), "std": float(s) if s > 0 else 1.0}

    # Apply standardization
    for feat in NUMERIC_FEATURES:
        df[feat] = (df[feat] - FEATURE_STATS[feat]["mean"]) / FEATURE_STATS[feat]["std"]

    features_list = [torch.tensor(row, dtype=torch.float32) for row in df.values]
    
    try:
        labels_list = [torch.tensor(int(val or 0), dtype=torch.long) for val in examples.get("is_fraud", [0])]
    except:
        labels_list = [torch.tensor(0, dtype=torch.long)] * batch_size

    return {"features": features_list, "label": labels_list}

def get_fraud_transforms():
    return {"train": fraud_transform, "test": fraud_transform}

def preprocess_transform(train, test):
    global BEHAVIORAL_LOOKUP, FEATURE_STATS
    
    # 1. Clear previous state to ensure clean preprocessing
    BEHAVIORAL_LOOKUP = {}
    FEATURE_STATS = {}
    
    # 2. Build combined lookup for both train and test to ensure behavioral features 
    # are calculated with full historical context (velocity, rolling windows)
    print("Building behavioral lookup for combined train and test data...")
    combined = pd.concat([train, test], axis=0)
    build_behavioral_lookup(combined)
    
    # 3. Transform both sets
    # We want the numeric values, not tensors, for CSV saving
    train_dic = fraud_transform(train)
    test_dic = fraud_transform(test)
    
    # Convert list of tensors back to numeric DataFrame
    # Note: features_list in fraud_transform contains 1D tensors
    train_features = np.array([t.numpy() for t in train_dic["features"]])
    test_features = np.array([t.numpy() for t in test_dic["features"]])
    
    train_df = pd.DataFrame(train_features, columns=NUMERIC_FEATURES)
    train_df["is_fraud"] = [int(t.item()) for t in train_dic["label"]]
    
    test_df = pd.DataFrame(test_features, columns=NUMERIC_FEATURES)
    test_df["is_fraud"] = [int(t.item()) for t in test_dic["label"]]
    
    return train_df, test_df

def processed_fraud_transform(examples):
    """Transform for data that is already processed (standardized and engineered)."""
    # Use torch.tensor on lists directly for efficiency
    # features is (batch_size, 12)
    feature_cols = [torch.tensor(examples[feat], dtype=torch.float32) for feat in NUMERIC_FEATURES]
    features = torch.stack(feature_cols, dim=1)
    
    # labels is (batch_size,)
    labels = torch.tensor(examples["is_fraud"], dtype=torch.long)
    
    return {"features": features, "label": labels}

def get_processed_fraud_transforms():
    """Return transforms specifically for pre-processed CSV data."""
    return {"train": processed_fraud_transform, "test": processed_fraud_transform}