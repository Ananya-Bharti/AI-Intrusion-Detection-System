"""
train.py — Train ML models for intrusion detection and save them.
Models: Random Forest, MLP Classifier, Isolation Forest
"""

import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Add parent dir so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

def get_models_dir():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'models')

def evaluate_model(name, y_true, y_pred, labels=None):
    """Print evaluation metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"  📊 {name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"{'='*60}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

def train_all():
    """Train all models and save them."""
    print("🔄 Loading and preprocessing data...")
    (X_train, y_train_bin, y_train_multi, train_cats,
     X_test, y_test_bin, y_test_multi, test_cats,
     feature_cols, label_encoders) = preprocess()
    
    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)
    metrics = {}
    
    # ──────────────────────────────────────────────────
    # 1. Random Forest (Binary Classification)
    # ──────────────────────────────────────────────────
    print("\n🌳 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_bin)
    rf_pred = rf.predict(X_test)
    metrics['random_forest'] = evaluate_model("Random Forest (Binary)", y_test_bin, rf_pred)
    joblib.dump(rf, os.path.join(models_dir, 'random_forest.pkl'))
    print("  ✅ Saved to models/random_forest.pkl")
    
    # ──────────────────────────────────────────────────
    # 2. Random Forest Multi-class
    # ──────────────────────────────────────────────────
    print("\n🌳 Training Random Forest (Multi-class)...")
    rf_multi = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_multi.fit(X_train, y_train_multi)
    rf_multi_pred = rf_multi.predict(X_test)
    metrics['rf_multi'] = evaluate_model("Random Forest (Multi-class)", y_test_multi, rf_multi_pred)
    joblib.dump(rf_multi, os.path.join(models_dir, 'random_forest_multi.pkl'))
    print("  ✅ Saved to models/random_forest_multi.pkl")
    
    # ──────────────────────────────────────────────────
    # 3. MLP Classifier (Binary)
    # ──────────────────────────────────────────────────
    print("\n🧠 Training MLP Classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        verbose=False
    )
    mlp.fit(X_train, y_train_bin)
    mlp_pred = mlp.predict(X_test)
    metrics['mlp'] = evaluate_model("MLP Classifier (Binary)", y_test_bin, mlp_pred)
    joblib.dump(mlp, os.path.join(models_dir, 'mlp_model.pkl'))
    print("  ✅ Saved to models/mlp_model.pkl")
    
    # ──────────────────────────────────────────────────
    # 4. Isolation Forest (Anomaly Detection)
    # ──────────────────────────────────────────────────
    print("\n🔍 Training Isolation Forest...")
    # Train only on normal traffic
    normal_mask = y_train_bin == 0
    X_train_normal = X_train[normal_mask]
    
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train_normal)
    
    # Predict: Isolation Forest returns 1 for inliers, -1 for outliers
    iso_pred_raw = iso.predict(X_test)
    # Convert: inlier(1) → normal(0), outlier(-1) → attack(1)
    iso_pred = np.where(iso_pred_raw == 1, 0, 1)
    metrics['isolation_forest'] = evaluate_model("Isolation Forest (Anomaly)", y_test_bin, iso_pred)
    joblib.dump(iso, os.path.join(models_dir, 'isolation_forest.pkl'))
    print("  ✅ Saved to models/isolation_forest.pkl")
    
    # Save feature importances from Random Forest
    feature_importance = dict(zip(feature_cols, rf.feature_importances_))
    joblib.dump(feature_importance, os.path.join(models_dir, 'feature_importance.pkl'))
    
    # Save metrics
    joblib.dump(metrics, os.path.join(models_dir, 'metrics.pkl'))
    
    print("\n" + "="*60)
    print("  🎉 All models trained and saved successfully!")
    print("="*60)
    
    return metrics

if __name__ == '__main__':
    train_all()
