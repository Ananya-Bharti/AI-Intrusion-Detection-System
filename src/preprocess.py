"""
preprocess.py — Load, clean, encode, and scale the NSL-KDD dataset.
Outputs clean DataFrames ready for ML model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

# NSL-KDD column names (41 features + label + difficulty_level)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# Attack type mapping to categories
ATTACK_MAP = {
    'normal': 'normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'snmpgetattack': 'R2L', 'named': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'sendmail': 'R2L', 'httptunnel': 'R2L', 'worm': 'R2L', 'snmpguess': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'xterm': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
}

def get_base_dir():
    """Get the base project directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    """Load raw NSL-KDD train and test datasets."""
    base = get_base_dir()
    train_path = os.path.join(base, 'data', 'KDDTrain+.txt')
    test_path = os.path.join(base, 'data', 'KDDTest+.txt')

    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)

    return train_df, test_df

def map_labels(df):
    """Map attack labels to binary and multi-class categories."""
    # Multi-class: map specific attacks to categories
    df['attack_category'] = df['label'].map(ATTACK_MAP).fillna('Unknown')
    
    # Binary: normal=0, attack=1
    df['binary_label'] = (df['attack_category'] != 'normal').astype(int)
    
    # Multi-class numeric: normal=0, DoS=1, Probe=2, R2L=3, U2R=4
    category_map = {'normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
    df['multi_label'] = df['attack_category'].map(category_map).fillna(1).astype(int)
    
    return df

def encode_and_scale(train_df, test_df):
    """Encode categorical features and scale numeric features."""
    base = get_base_dir()
    
    # Drop difficulty_level (not a network feature)
    train_df = train_df.drop('difficulty_level', axis=1)
    test_df = test_df.drop('difficulty_level', axis=1)
    
    # Map labels first
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined unique values from both train and test
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le
    
    # Extract labels before scaling
    train_binary = train_df['binary_label'].values
    test_binary = test_df['binary_label'].values
    train_multi = train_df['multi_label'].values
    test_multi = test_df['multi_label'].values
    train_categories = train_df['attack_category'].values
    test_categories = test_df['attack_category'].values
    
    # Feature columns (exclude label columns)
    feature_cols = [c for c in train_df.columns 
                    if c not in ['label', 'binary_label', 'multi_label', 'attack_category']]
    
    X_train = train_df[feature_cols].values.astype(np.float64)
    X_test = test_df[feature_cols].values.astype(np.float64)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler and encoders
    models_dir = os.path.join(base, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.pkl'))
    
    print(f"✅ Preprocessing complete!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples:     {X_test.shape[0]}")
    print(f"   Features:         {X_train.shape[1]}")
    print(f"   Scaler saved to models/scaler.pkl")
    
    return (X_train, train_binary, train_multi, train_categories,
            X_test, test_binary, test_multi, test_categories,
            feature_cols, label_encoders)

def preprocess():
    """Full preprocessing pipeline."""
    train_df, test_df = load_data()
    return encode_and_scale(train_df, test_df)

if __name__ == '__main__':
    preprocess()
