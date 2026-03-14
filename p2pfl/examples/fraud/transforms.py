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
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Features to use: 
# 0: amt, 1: lat, 2: long, 3: city_pop, 4: merch_lat, 5: merch_long, 
# 6: distance, 7: hour, 8: day_of_week, 9: category_idx, 10: age
NUMERIC_FEATURES = [
    "amt", "lat", "long", "city_pop", "merch_lat", "merch_long", 
    "distance", "hour", "day_of_week", "category_idx", "age"
]

# Mapping for categories (Top categories from dataset)
CATEGORY_MAP = {
    'misc_net': 0, 'grocery_pos': 1, 'entertainment': 2, 'gas_transport': 3,
    'misc_pos': 4, 'grocery_net': 5, 'shopping_net': 6, 'shopping_pos': 7,
    'food_dining': 8, 'personal_care': 9, 'health_fitness': 10, 'travel': 11,
    'kids_pets': 12, 'home': 13
}

# Feature stats for standardization
FEATURE_STATS = {}

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
    except: return 40.0 # Default age

def fraud_transform(examples):
    """Transform batch into 11 high-quality features."""
    global FEATURE_STATS
    batch_size = len(examples.get("amt", []))
    
    data_dict = {feat: [] for feat in NUMERIC_FEATURES}
    
    for idx in range(batch_size):
        # 1. Raw numeric
        for f in ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]:
            data_dict[f].append(float(examples.get(f, [0])[idx] or 0))
        
        # 2. Distance
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
            data_dict["hour"].append(0.0)
            data_dict["day_of_week"].append(0.0)
            
        # 4. Categorical
        cat = examples.get("category", [""])[idx]
        data_dict["category_idx"].append(float(CATEGORY_MAP.get(cat, 14)))
        
        # 5. Age
        data_dict["age"].append(calculate_age(examples.get("dob", ["1980-01-01"])[idx]))

    df = pd.DataFrame(data_dict)
    
    # Robust Auto-init
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
