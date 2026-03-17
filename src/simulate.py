"""
simulate.py — Simulate network traffic streams (normal + attack).
Provides generators for the dashboard to mimic live traffic.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import COLUMN_NAMES, ATTACK_MAP, load_data, map_labels

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _prepare_test_data():
    """Load and prepare test data for simulation."""
    base = get_base_dir()
    test_path = os.path.join(base, 'data', 'KDDTest+.txt')
    
    df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    df = df.drop('difficulty_level', axis=1)
    df = map_labels(df)
    
    return df

def _encode_row(row, label_encoders, scaler, feature_cols):
    """Encode and scale a single row for prediction."""
    row_copy = row.copy()
    
    for col, le in label_encoders.items():
        if col in row_copy.index:
            val = row_copy[col]
            if val in le.classes_:
                row_copy[col] = le.transform([val])[0]
            else:
                row_copy[col] = 0  # Unknown category
    
    # Get feature values
    features = []
    for col in feature_cols:
        if col in row_copy.index:
            features.append(float(row_copy[col]))
        else:
            features.append(0.0)
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    return features_scaled

def get_stream_data():
    """Load all necessary data and models for streaming."""
    base = get_base_dir()
    models_dir = os.path.join(base, 'models')
    
    df = _prepare_test_data()
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
    
    return df, scaler, label_encoders, feature_cols

def normal_stream(df, n=10):
    """Get n random normal (safe) traffic samples."""
    normal_df = df[df['attack_category'] == 'normal']
    if len(normal_df) == 0:
        return pd.DataFrame()
    samples = normal_df.sample(n=min(n, len(normal_df)), replace=True)
    return samples

def attack_stream(df, attack_type='DoS', n=10):
    """Get n random samples of a specific attack category."""
    attack_df = df[df['attack_category'] == attack_type]
    if len(attack_df) == 0:
        return pd.DataFrame()
    samples = attack_df.sample(n=min(n, len(attack_df)), replace=True)
    return samples

def mixed_stream(df, n=20, attack_ratio=0.3, attack_type='DoS'):
    """Get a mix of normal and attack traffic."""
    n_attack = max(1, int(n * attack_ratio))
    n_normal = n - n_attack
    
    normal_samples = normal_stream(df, n_normal)
    attack_samples = attack_stream(df, attack_type, n_attack)
    
    mixed = pd.concat([normal_samples, attack_samples], ignore_index=True)
    mixed = mixed.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    return mixed

def get_single_sample(df, attack_type=None):
    """Get a single sample, either normal or of a specific attack type."""
    if attack_type is None or attack_type == 'normal':
        subset = df[df['attack_category'] == 'normal']
    else:
        subset = df[df['attack_category'] == attack_type]
    
    if len(subset) == 0:
        return df.sample(n=1).iloc[0]
    
    return subset.sample(n=1).iloc[0]


def realistic_mixed_stream(df, n=25, attack_ratio=0.35):
    """
    Simulate realistic network traffic with ALL attack types mixed randomly.
    The model decides what each packet is — user does not pre-select attack type.

    Attack type distribution mimics a real network incident:
      DoS   ~50% of attack traffic (volumetric, most common)
      Probe ~30% of attack traffic (scanning activity)
      R2L   ~15% of attack traffic (credential exploits)
      U2R   ~5%  of attack traffic (privilege escalation, rarest)
    """
    n_attack = max(1, int(n * attack_ratio))
    n_normal = n - n_attack

    # Proportional sampling across all four attack types
    attack_proportions = {'DoS': 0.50, 'Probe': 0.30, 'R2L': 0.15, 'U2R': 0.05}
    attack_frames = []
    for atype, proportion in attack_proportions.items():
        count = max(1, round(n_attack * proportion))
        subset = df[df['attack_category'] == atype]
        if len(subset) > 0:
            attack_frames.append(subset.sample(n=min(count, len(subset)), replace=True))

    normal_samples = normal_stream(df, n_normal)
    all_frames = [normal_samples] + attack_frames
    mixed = pd.concat(all_frames, ignore_index=True)
    mixed = mixed.sample(frac=1).reset_index(drop=True)  # Shuffle for realism
    return mixed.head(n)


def normal_burst(df, n=25):
    """Get a burst of purely normal traffic (network idle / monitoring mode)."""
    return normal_stream(df, n)
