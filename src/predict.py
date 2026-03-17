"""
predict.py — Run predictions on single samples using trained ML models.
"""

import numpy as np
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

CATEGORY_NAMES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

def load_model(model_name='random_forest'):
    """Load a trained model by name."""
    models_dir = get_models_dir()
    
    model_files = {
        'random_forest': 'random_forest.pkl',
        'random_forest_multi': 'random_forest_multi.pkl',
        'mlp': 'mlp_model.pkl',
        'isolation_forest': 'isolation_forest.pkl',
    }
    
    filename = model_files.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    return joblib.load(os.path.join(models_dir, filename))

def predict_single(features_scaled, model_name='random_forest'):
    """
    Predict on a single scaled sample.
    
    Args:
        features_scaled: numpy array of shape (1, n_features), already scaled
        model_name: which model to use
    
    Returns:
        dict with label, confidence, attack_type
    """
    model = load_model(model_name)
    
    if model_name == 'isolation_forest':
        # Isolation Forest: 1=inlier(normal), -1=outlier(attack)
        raw_pred = model.predict(features_scaled)[0]
        score = model.score_samples(features_scaled)[0]
        
        is_attack = raw_pred == -1
        # Convert score to a 0-1 confidence (more negative = more anomalous)
        confidence = min(1.0, max(0.0, -score))
        
        return {
            'label': 'Attack' if is_attack else 'Normal',
            'binary': 1 if is_attack else 0,
            'confidence': confidence if is_attack else 1.0 - confidence,
            'attack_type': 'Anomaly' if is_attack else 'Normal',
            'model': 'Isolation Forest'
        }
    
    elif model_name == 'random_forest_multi':
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        attack_type = CATEGORY_NAMES.get(pred, 'Unknown')
        
        return {
            'label': 'Attack' if pred != 0 else 'Normal',
            'binary': 0 if pred == 0 else 1,
            'confidence': confidence,
            'attack_type': attack_type,
            'model': 'Random Forest (Multi-class)',
            'class_probabilities': {CATEGORY_NAMES[i]: float(p) for i, p in enumerate(proba)}
        }
    
    else:
        # Binary classifiers (RF, MLP)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        
        model_display = {
            'random_forest': 'Random Forest',
            'mlp': 'MLP Neural Network'
        }.get(model_name, model_name)
        
        return {
            'label': 'Attack' if pred == 1 else 'Normal',
            'binary': int(pred),
            'confidence': confidence,
            'attack_type': 'Attack' if pred == 1 else 'Normal',
            'model': model_display
        }

def predict_batch(features_scaled, model_name='random_forest'):
    """Predict on a batch of scaled features."""
    results = []
    for i in range(features_scaled.shape[0]):
        result = predict_single(features_scaled[i:i+1], model_name)
        results.append(result)
    return results
