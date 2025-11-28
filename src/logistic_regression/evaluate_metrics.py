# Task 2)b

import numpy as np
import pickle
import os
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')

RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
MODELS_PATH = os.path.join(RESULTS_FOLDER, 'trained_models.pkl')
SPLIT_DATA_PATH = os.path.join(RESULTS_FOLDER, 'split_data.pkl')
OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'metrics')

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("METRICS EVALUATION - TASK 2(b)")
print("=" * 70)

# ==========================================
# 1. LOAD TRAINED MODELS AND SPLIT DATA
# ==========================================
print("\n--- 1. LOADING MODELS AND DATA ---")

if not os.path.exists(MODELS_PATH):
    print(f"Error: Trained models not found at {MODELS_PATH}")
    print("Please run train_logistic_regression.py first to train models.")
    exit(1)

if not os.path.exists(SPLIT_DATA_PATH):
    print(f"Error: Split data not found at {SPLIT_DATA_PATH}")
    print("Please run train_logistic_regression.py first.")
    exit(1)

# Load trained models
with open(MODELS_PATH, 'rb') as f:
    trained_models = pickle.load(f)

# Load split data
with open(SPLIT_DATA_PATH, 'rb') as f:
    split_data = pickle.load(f)

print(f"Loaded {len(trained_models)} trained models")
print(f"Feature sets: {list(trained_models.keys())}")

# Extract split data
indices_train = split_data['indices_train']
indices_test = split_data['indices_test']
y_train = split_data['y_train']
y_test = split_data['y_test']
feature_sets_data = split_data['feature_sets']

# ==========================================
# 2. HELPER FUNCTION: CALCULATE METRICS
# ==========================================

def calculate_metrics(y_true, y_pred, y_proba=None, classes=None):
    """
    Calculate classification metrics for multiclass problem.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (n_samples, n_classes) - optional for AUC
        classes: List of class labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro and weighted averages)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = dict(zip(classes, precision_per_class))
    metrics['recall_per_class'] = dict(zip(classes, recall_per_class))
    metrics['f1_per_class'] = dict(zip(classes, f1_per_class))
    
    # AUC (multiclass: one-vs-rest)
    if y_proba is not None and classes is not None:
        try:
            # Encode labels to integers for AUC calculation
            le = LabelEncoder()
            y_true_encoded = le.fit_transform(y_true)
            
            # Calculate AUC using one-vs-rest approach
            # y_proba should be (n_samples, n_classes) with probabilities for each class
            if y_proba.shape[1] == len(classes):
                # Use macro average for multiclass AUC
                metrics['auc_macro'] = roc_auc_score(
                    y_true_encoded, y_proba, 
                    multi_class='ovr', 
                    average='macro'
                )
            else:
                metrics['auc_macro'] = None
        except Exception as e:
            print(f"  Warning: Could not calculate AUC: {e}")
            metrics['auc_macro'] = None
    else:
        metrics['auc_macro'] = None
    
    return metrics

# ==========================================
# 3. EVALUATE ON TRAINING SET
# ==========================================
print("\n--- 2. EVALUATING ON TRAINING SET ---")

train_results = {}

for feature_type, model_data in trained_models.items():
    model = model_data['model']
    feature_name = model_data['name']
    
    print(f"\nEvaluating: {feature_name}")
    
    # Get training data for this feature set
    X_train = feature_sets_data[feature_type]['X_train']
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_train, y_train_pred, y_train_proba, 
        classes=model.classes
    )
    
    train_results[feature_type] = {
        'name': feature_name,
        'metrics': metrics,
        'y_pred': y_train_pred,
        'y_proba': y_train_proba
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f} (macro), {metrics['precision_weighted']:.4f} (weighted)")
    print(f"  Recall:    {metrics['recall_macro']:.4f} (macro), {metrics['recall_weighted']:.4f} (weighted)")
    print(f"  F1-score:  {metrics['f1_macro']:.4f} (macro), {metrics['f1_weighted']:.4f} (weighted)")
    if metrics['auc_macro'] is not None:
        print(f"  AUC:       {metrics['auc_macro']:.4f}")

# ==========================================
# 4. EVALUATE ON TEST SET
# ==========================================
print("\n--- 3. EVALUATING ON TEST SET ---")

test_results = {}

for feature_type, model_data in trained_models.items():
    model = model_data['model']
    feature_name = model_data['name']
    
    print(f"\nEvaluating: {feature_name}")
    
    # Get test data for this feature set
    X_test = feature_sets_data[feature_type]['X_test']
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_test, y_test_pred, y_test_proba,
        classes=model.classes
    )
    
    test_results[feature_type] = {
        'name': feature_name,
        'metrics': metrics,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f} (macro), {metrics['precision_weighted']:.4f} (weighted)")
    print(f"  Recall:    {metrics['recall_macro']:.4f} (macro), {metrics['recall_weighted']:.4f} (weighted)")
    print(f"  F1-score:  {metrics['f1_macro']:.4f} (macro), {metrics['f1_weighted']:.4f} (weighted)")
    if metrics['auc_macro'] is not None:
        print(f"  AUC:       {metrics['auc_macro']:.4f}")

# ==========================================
# 5. CREATE COMPARISON TABLES
# ==========================================
print("\n--- 4. CREATING COMPARISON TABLES ---")

# Training set comparison
train_df = pd.DataFrame({
    'Feature Set': [train_results[ft]['name'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Accuracy': [train_results[ft]['metrics']['accuracy'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Precision (Macro)': [train_results[ft]['metrics']['precision_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Recall (Macro)': [train_results[ft]['metrics']['recall_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'F1-Score (Macro)': [train_results[ft]['metrics']['f1_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'AUC (Macro)': [train_results[ft]['metrics']['auc_macro'] if train_results[ft]['metrics']['auc_macro'] is not None else np.nan 
                    for ft in ['images', 'metadata', 'text', 'fused']]
})

# Test set comparison
test_df = pd.DataFrame({
    'Feature Set': [test_results[ft]['name'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Accuracy': [test_results[ft]['metrics']['accuracy'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Precision (Macro)': [test_results[ft]['metrics']['precision_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Recall (Macro)': [test_results[ft]['metrics']['recall_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'F1-Score (Macro)': [test_results[ft]['metrics']['f1_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'AUC (Macro)': [test_results[ft]['metrics']['auc_macro'] if test_results[ft]['metrics']['auc_macro'] is not None else np.nan 
                    for ft in ['images', 'metadata', 'text', 'fused']]
})

# Combined comparison (Training vs Test)
comparison_df = pd.DataFrame({
    'Feature Set': [train_results[ft]['name'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Train Accuracy': [train_results[ft]['metrics']['accuracy'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Test Accuracy': [test_results[ft]['metrics']['accuracy'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Train F1 (Macro)': [train_results[ft]['metrics']['f1_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Test F1 (Macro)': [test_results[ft]['metrics']['f1_macro'] for ft in ['images', 'metadata', 'text', 'fused']],
    'Train AUC': [train_results[ft]['metrics']['auc_macro'] if train_results[ft]['metrics']['auc_macro'] is not None else np.nan 
                  for ft in ['images', 'metadata', 'text', 'fused']],
    'Test AUC': [test_results[ft]['metrics']['auc_macro'] if test_results[ft]['metrics']['auc_macro'] is not None else np.nan 
                 for ft in ['images', 'metadata', 'text', 'fused']]
})

# Print tables
print("\n" + "=" * 70)
print("TRAINING SET METRICS")
print("=" * 70)
print(train_df.to_string(index=False))
print()

print("=" * 70)
print("TEST SET METRICS")
print("=" * 70)
print(test_df.to_string(index=False))
print()

print("=" * 70)
print("TRAINING vs TEST COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))
print()

# Save tables to CSV
train_df.to_csv(os.path.join(OUTPUT_FOLDER, 'train_metrics.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_FOLDER, 'test_metrics.csv'), index=False)
comparison_df.to_csv(os.path.join(OUTPUT_FOLDER, 'comparison_metrics.csv'), index=False)

print(f"\nMetrics tables saved to: {OUTPUT_FOLDER}")

# ==========================================
# 6. DETAILED PER-CLASS METRICS
# ==========================================
print("\n--- 5. PER-CLASS METRICS (TEST SET) ---")

for feature_type in ['images', 'metadata', 'text', 'fused']:
    feature_name = test_results[feature_type]['name']
    metrics = test_results[feature_type]['metrics']
    
    print(f"\n{feature_name}:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for cls in metrics['precision_per_class'].keys():
        prec = metrics['precision_per_class'][cls]
        rec = metrics['recall_per_class'][cls]
        f1 = metrics['f1_per_class'][cls]
        print(f"{cls:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

# ==========================================
# 7. CONFUSION MATRICES (TEST SET)
# ==========================================
print("\n--- 6. CONFUSION MATRICES (TEST SET) ---")

for feature_type in ['images', 'metadata', 'text', 'fused']:
    feature_name = test_results[feature_type]['name']
    y_pred = test_results[feature_type]['y_pred']
    
    class_labels = list(test_results[feature_type]['metrics']['precision_per_class'].keys())
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    
    print(f"\n{feature_name}:")
    print("Confusion Matrix:")
    print(cm)
    print()

# ==========================================
# 8. SUMMARY AND RANKING
# ==========================================
print("\n" + "=" * 70)
print("SUMMARY AND RANKING")
print("=" * 70)

# Rank by test accuracy
test_accuracies = [(ft, test_results[ft]['metrics']['accuracy']) for ft in ['images', 'metadata', 'text', 'fused']]
test_accuracies.sort(key=lambda x: x[1], reverse=True)

print("\nRanking by Test Accuracy:")
for rank, (ft, acc) in enumerate(test_accuracies, 1):
    print(f"  {rank}. {test_results[ft]['name']}: {acc:.4f}")

# Rank by test F1-score
test_f1s = [(ft, test_results[ft]['metrics']['f1_macro']) for ft in ['images', 'metadata', 'text', 'fused']]
test_f1s.sort(key=lambda x: x[1], reverse=True)

print("\nRanking by Test F1-Score (Macro):")
for rank, (ft, f1) in enumerate(test_f1s, 1):
    print(f"  {rank}. {test_results[ft]['name']}: {f1:.4f}")

print("\n" + "=" * 70)
print("METRICS EVALUATION COMPLETED")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_FOLDER}")
print("  - train_metrics.csv")
print("  - test_metrics.csv")
print("  - comparison_metrics.csv")