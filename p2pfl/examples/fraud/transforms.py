#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#

"""Transform functions for fraud detection dataset with online behavioral engineering."""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Features to use (14 total)
NUMERIC_FEATURES = [
    "amt", "lat", "long", "city_pop", "merch_lat", "merch_long", 
    "distance", "hour", "day_of_week", "category_idx", "age", "unix_time",
    "amt_diff_avg_30d", "trans_count_24h"
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
    """Pre-calculate behavioral features for the local partition using pandas rolling windows."""
    global BEHAVIORAL_LOOKUP
    df = pd.DataFrame(examples)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='mixed')
    df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])
    
    # 1. Rolling Average Amount (30 days window)
    temp_df = df.set_index('trans_date_trans_time')
    df['avg_amt_30d'] = temp_df.groupby('cc_num')['amt'].transform(
        lambda x: x.rolling(window='30D', min_periods=1).mean()
    ).values
    
    # 2. Transaction Count (24h window)
    df['trans_count_24h'] = temp_df.groupby('cc_num')['amt'].transform(
        lambda x: x.rolling(window='24H', min_periods=1).count()
    ).values
    
    # Update global lookup table
    for _, row in df.iterrows():
        key = (row['cc_num'], int(row['unix_time']))
        BEHAVIORAL_LOOKUP[key] = {
            'amt_diff_avg_30d': float(row['amt'] - row['avg_amt_30d']),
            'trans_count_24h': float(row['trans_count_24h'])
        }

def fraud_transform(examples):
    """Transform batch using advanced features and lazy lookup."""
    global FEATURE_STATS, BEHAVIORAL_LOOKUP
    batch_size = len(examples.get("amt", []))
    
    # Initialize behavioral lookup once per Node lifecycle
    if not BEHAVIORAL_LOOKUP and batch_size > 1:
        build_behavioral_lookup(examples)
    
    data_dict = {feat: [] for feat in NUMERIC_FEATURES}
    
    for idx in range(batch_size):
        # 1. Raw numeric
        for f in ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "unix_time"]:
            data_dict[f].append(float(examples.get(f, [0])[idx] or 0))
        
        # 2. Calculated Distance
        dist = haversine(examples["lat"][idx], examples["long"][idx], 
                         examples["merch_lat"][idx], examples["merch_long"][idx])
        data_dict["distance"].append(dist)
        
        # 3. Temporal
        try:
            dt_str = examples["trans_date_trans_time"][idx]
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            data_dict["hour"].append(float(dt.hour))
            data_dict["day_of_week"].append(float(dt.weekday()))
        except:
            data_dict["hour"].append(0.0); data_dict["day_of_week"].append(0.0)
            
        # 4. Categorical & Age
        cat = examples.get("category", [""])[idx]
        data_dict["category_idx"].append(float(CATEGORY_MAP.get(cat, 14)))
        data_dict["age"].append(calculate_age(examples.get("dob", ["1980-01-01"])[idx]))
        
        # 5. Behavioral Lookup (Online tra cứu từ bảng đã dựng)
        key = (examples['cc_num'][idx], int(examples['unix_time'][idx]))
        beh = BEHAVIORAL_LOOKUP.get(key, {'amt_diff_avg_30d': 0.0, 'trans_count_24h': 1.0})
        data_dict['amt_diff_avg_30d'].append(beh['amt_diff_avg_30d'])
        data_dict['trans_count_24h'].append(beh['trans_count_24h'])

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
