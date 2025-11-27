"""
Compare our implementation with Scikit-learn's LogisticRegression
Task 2(c): Compare performance and runtime with Scikit-learn implementation
"""

import numpy as np
import pickle
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')

# Add src to path for pickle loading
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
MODELS_PATH = os.path.join(RESULTS_FOLDER, 'trained_models.pkl')
SPLIT_DATA_PATH = os.path.join(RESULTS_FOLDER, 'split_data.pkl')
OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'comparison')
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'plots')

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Model hyperparameters (from train_logistic_regression.py)
LEARNING_RATE = 0.1
MAX_ITER = 1000
TOLERANCE = 1e-6
REGULARIZATION = 'l2'
LAMBDA_REG = 0.1

print("=" * 70)
print("COMPARISON WITH SCIKIT-LEARN - TASK 2(c)")
print("=" * 70)

# ==========================================
# 1. LOAD DATA AND MODELS
# ==========================================
print("\n--- 1. LOADING DATA AND MODELS ---")

if not os.path.exists(MODELS_PATH):
    print(f"Error: Trained models not found at {MODELS_PATH}")
    print("Please run train_logistic_regression.py first to train models.")
    exit(1)

if not os.path.exists(SPLIT_DATA_PATH):
    print(f"Error: Split data not found at {SPLIT_DATA_PATH}")
    print("Please run train_logistic_regression.py first.")
    exit(1)

# Load trained models (our implementation)
with open(MODELS_PATH, 'rb') as f:
    our_models = pickle.load(f)

# Load split data
with open(SPLIT_DATA_PATH, 'rb') as f:
    split_data = pickle.load(f)

print(f"Loaded {len(our_models)} trained models (our implementation)")
print(f"Feature sets: {list(our_models.keys())}")

# Extract split data
y_train = split_data['y_train']
y_test = split_data['y_test']
y_val = split_data['y_val']

# ==========================================
# 2. TRAIN SCIKIT-LEARN MODELS
# ==========================================
print("\n--- 2. TRAINING SCIKIT-LEARN MODELS ---")

# Convert regularization parameter
# Scikit-learn uses C = 1 / (lambda * n_samples) for L2 regularization
# We'll use C = 1 / lambda_reg to approximate
C_value = 1.0 / LAMBDA_REG if LAMBDA_REG > 0 else 1.0

# Scikit-learn uses tol (tolerance) instead of tolerance
tol_value = TOLERANCE

# Scikit-learn uses max_iter instead of max_iter (same name)
max_iter_value = MAX_ITER

# Note: Scikit-learn doesn't have a direct learning_rate parameter for LogisticRegression
# It uses different solvers. We'll use 'lbfgs' which is default and works well.

sklearn_models = {}
sklearn_training_times = {}

for feature_type in ['images', 'metadata', 'text', 'fused']:
    if feature_type not in our_models:
        continue
    
    print(f"\nTraining Scikit-learn model for: {feature_type}")
    
    # Get feature sets from split_data
    feature_sets = split_data['feature_sets']
    if feature_type not in feature_sets:
        print(f"Warning: {feature_type} not found in split_data")
        continue
    
    X_train = feature_sets[feature_type]['X_train']
    X_test = feature_sets[feature_type]['X_test']
    X_val = feature_sets[feature_type]['X_val']
    
    # Create Scikit-learn model
    # Use OneVsRestClassifier for multiclass
    base_lr = SklearnLR(
        C=C_value,
        max_iter=max_iter_value,
        tol=tol_value,
        penalty=REGULARIZATION if REGULARIZATION else 'none',
        solver='lbfgs',  # Works well for small datasets
        random_state=42,
        verbose=0
    )
    
    sklearn_model = OneVsRestClassifier(base_lr)
    
    # Train and measure time
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    sklearn_models[feature_type] = {
        'model': sklearn_model,
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'training_time': training_time
    }
    
    sklearn_training_times[feature_type] = training_time
    print(f"  Training completed in {training_time:.4f} seconds")

# ==========================================
# 3. EVALUATE PERFORMANCE
# ==========================================
print("\n--- 3. EVALUATING PERFORMANCE ---")

comparison_results = []

for feature_type in ['images', 'metadata', 'text', 'fused']:
    if feature_type not in our_models or feature_type not in sklearn_models:
        continue
    
    print(f"\nEvaluating: {feature_type}")
    
    # Get data
    X_test = sklearn_models[feature_type]['X_test']
    X_train = sklearn_models[feature_type]['X_train']
    
    # Our implementation predictions
    our_model = our_models[feature_type]['model']
    
    # Training predictions
    start_time = time.time()
    our_train_pred = our_model.predict(X_train)
    our_train_time = time.time() - start_time
    
    start_time = time.time()
    our_test_pred = our_model.predict(X_test)
    our_test_time = time.time() - start_time
    
    # Test predictions
    our_train_proba = our_model.predict_proba(X_train)
    our_test_proba = our_model.predict_proba(X_test)
    
    # Scikit-learn predictions
    sklearn_model = sklearn_models[feature_type]['model']
    
    start_time = time.time()
    sklearn_train_pred = sklearn_model.predict(X_train)
    sklearn_train_time = time.time() - start_time
    
    start_time = time.time()
    sklearn_test_pred = sklearn_model.predict(X_test)
    sklearn_test_time = time.time() - start_time
    
    sklearn_train_proba = sklearn_model.predict_proba(X_train)
    sklearn_test_proba = sklearn_model.predict_proba(X_test)
    
    # Calculate metrics for our implementation
    our_train_acc = accuracy_score(y_train, our_train_pred)
    our_test_acc = accuracy_score(y_test, our_test_pred)
    our_train_prec = precision_score(y_train, our_train_pred, average='weighted', zero_division=0)
    our_test_prec = precision_score(y_test, our_test_pred, average='weighted', zero_division=0)
    our_train_rec = recall_score(y_train, our_train_pred, average='weighted', zero_division=0)
    our_test_rec = recall_score(y_test, our_test_pred, average='weighted', zero_division=0)
    our_train_f1 = f1_score(y_train, our_train_pred, average='weighted', zero_division=0)
    our_test_f1 = f1_score(y_test, our_test_pred, average='weighted', zero_division=0)
    
    # Calculate metrics for Scikit-learn
    sklearn_train_acc = accuracy_score(y_train, sklearn_train_pred)
    sklearn_test_acc = accuracy_score(y_test, sklearn_test_pred)
    sklearn_train_prec = precision_score(y_train, sklearn_train_pred, average='weighted', zero_division=0)
    sklearn_test_prec = precision_score(y_test, sklearn_test_pred, average='weighted', zero_division=0)
    sklearn_train_rec = recall_score(y_train, sklearn_train_pred, average='weighted', zero_division=0)
    sklearn_test_rec = recall_score(y_test, sklearn_test_pred, average='weighted', zero_division=0)
    sklearn_train_f1 = f1_score(y_train, sklearn_train_pred, average='weighted', zero_division=0)
    sklearn_test_f1 = f1_score(y_test, sklearn_test_pred, average='weighted', zero_division=0)
    
    # Store results
    comparison_results.append({
        'feature_set': feature_type,
        'implementation': 'Our Implementation',
        'train_accuracy': our_train_acc,
        'test_accuracy': our_test_acc,
        'train_precision': our_train_prec,
        'test_precision': our_test_prec,
        'train_recall': our_train_rec,
        'test_recall': our_test_rec,
        'train_f1': our_train_f1,
        'test_f1': our_test_f1,
        'training_time': our_models[feature_type]['training_time'],
        'train_pred_time': our_train_time,
        'test_pred_time': our_test_time
    })
    
    comparison_results.append({
        'feature_set': feature_type,
        'implementation': 'Scikit-learn',
        'train_accuracy': sklearn_train_acc,
        'test_accuracy': sklearn_test_acc,
        'train_precision': sklearn_train_prec,
        'test_precision': sklearn_test_prec,
        'train_recall': sklearn_train_rec,
        'test_recall': sklearn_test_rec,
        'train_f1': sklearn_train_f1,
        'test_f1': sklearn_test_f1,
        'training_time': sklearn_models[feature_type]['training_time'],
        'train_pred_time': sklearn_train_time,
        'test_pred_time': sklearn_test_time
    })
    
    print(f"  Our Implementation - Train Acc: {our_train_acc:.4f}, Test Acc: {our_test_acc:.4f}")
    print(f"  Scikit-learn      - Train Acc: {sklearn_train_acc:.4f}, Test Acc: {sklearn_test_acc:.4f}")

# Convert to DataFrame
df_comparison = pd.DataFrame(comparison_results)

# Save results
df_comparison.to_csv(os.path.join(OUTPUT_FOLDER, 'comparison_results.csv'), index=False)
print(f"\n--- Results saved to {OUTPUT_FOLDER}/comparison_results.csv ---")

# ==========================================
# 4. CREATE VISUALIZATIONS
# ==========================================
print("\n--- 4. CREATING VISUALIZATIONS ---")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 4.1 Performance Comparison (Accuracy, Precision, Recall, F1)
print("  Creating performance comparison plots...")

metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in metrics:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(['train', 'test']):
        ax = axes[idx]
        
        # Prepare data
        metric_col = f'{dataset}_{metric}'
        data = df_comparison.pivot(index='feature_set', columns='implementation', values=metric_col)
        
        # Create bar plot
        x = np.arange(len(data.index))
        width = 0.35
        
        our_values = data['Our Implementation'].values
        sklearn_values = data['Scikit-learn'].values
        
        bars1 = ax.bar(x - width/2, our_values, width, label='Our Implementation', alpha=0.8)
        bars2 = ax.bar(x + width/2, sklearn_values, width, label='Scikit-learn', alpha=0.8)
        
        ax.set_xlabel('Feature Set', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Comparison - {dataset.capitalize()} Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, f'performance_comparison_{metric}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 4.2 Runtime Comparison
print("  Creating runtime comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Training time
ax = axes[0]
train_times = df_comparison.pivot(index='feature_set', columns='implementation', values='training_time')
x = np.arange(len(train_times.index))
width = 0.35
bars1 = ax.bar(x - width/2, train_times['Our Implementation'], width, label='Our Implementation', alpha=0.8)
bars2 = ax.bar(x + width/2, train_times['Scikit-learn'], width, label='Scikit-learn', alpha=0.8)
ax.set_xlabel('Feature Set', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(train_times.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}s',
               ha='center', va='bottom', fontsize=9)

# Prediction time (test)
ax = axes[1]
pred_times = df_comparison.pivot(index='feature_set', columns='implementation', values='test_pred_time')
x = np.arange(len(pred_times.index))
bars1 = ax.bar(x - width/2, pred_times['Our Implementation'], width, label='Our Implementation', alpha=0.8)
bars2 = ax.bar(x + width/2, pred_times['Scikit-learn'], width, label='Scikit-learn', alpha=0.8)
ax.set_xlabel('Feature Set', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Test Prediction Time Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pred_times.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}s',
               ha='center', va='bottom', fontsize=9)

# Speed ratio (Scikit-learn / Our Implementation)
ax = axes[2]
speed_ratios = {}
for feature_set in train_times.index:
    our_train_time = train_times.loc[feature_set, 'Our Implementation']
    sklearn_train_time = train_times.loc[feature_set, 'Scikit-learn']
    ratio = sklearn_train_time / our_train_time if our_train_time > 0 else 0
    speed_ratios[feature_set] = ratio

x = np.arange(len(speed_ratios))
bars = ax.bar(x, list(speed_ratios.values()), alpha=0.8, color='coral')
ax.set_xlabel('Feature Set', fontsize=12)
ax.set_ylabel('Speed Ratio', fontsize=12)
ax.set_title('Training Speed Ratio\n(Scikit-learn / Our Implementation)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(speed_ratios.keys()), rotation=45, ha='right')
ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Equal Speed')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i, (bar, ratio) in enumerate(zip(bars, speed_ratios.values())):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{ratio:.2f}x',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, 'runtime_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4.3 Performance Difference Heatmap
print("  Creating performance difference heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, dataset in enumerate(['train', 'test']):
    ax = axes[idx]
    
    # Calculate differences (Our - Scikit-learn)
    diff_data = []
    for feature_set in ['images', 'metadata', 'text', 'fused']:
        our_row = df_comparison[(df_comparison['feature_set'] == feature_set) & 
                               (df_comparison['implementation'] == 'Our Implementation')]
        sklearn_row = df_comparison[(df_comparison['feature_set'] == feature_set) & 
                                   (df_comparison['implementation'] == 'Scikit-learn')]
        
        if len(our_row) > 0 and len(sklearn_row) > 0:
            diff_data.append({
                'feature_set': feature_set,
                'accuracy': our_row[f'{dataset}_accuracy'].values[0] - sklearn_row[f'{dataset}_accuracy'].values[0],
                'precision': our_row[f'{dataset}_precision'].values[0] - sklearn_row[f'{dataset}_precision'].values[0],
                'recall': our_row[f'{dataset}_recall'].values[0] - sklearn_row[f'{dataset}_recall'].values[0],
                'f1': our_row[f'{dataset}_f1'].values[0] - sklearn_row[f'{dataset}_f1'].values[0]
            })
    
    diff_df = pd.DataFrame(diff_data)
    diff_df.set_index('feature_set', inplace=True)
    
    # Create heatmap
    sns.heatmap(diff_df.T, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Difference (Our - Scikit-learn)'})
    ax.set_title(f'Performance Difference - {dataset.capitalize()} Set\n(Positive = Our Implementation Better)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, 'performance_difference_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4.4 Summary Comparison Table Visualization
print("  Creating summary comparison table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create summary table
summary_data = []
for feature_set in ['images', 'metadata', 'text', 'fused']:
    our_row = df_comparison[(df_comparison['feature_set'] == feature_set) & 
                           (df_comparison['implementation'] == 'Our Implementation')]
    sklearn_row = df_comparison[(df_comparison['feature_set'] == feature_set) & 
                               (df_comparison['implementation'] == 'Scikit-learn')]
    
    if len(our_row) > 0 and len(sklearn_row) > 0:
        summary_data.append([
            feature_set.capitalize(),
            f"{our_row['test_accuracy'].values[0]:.4f}",
            f"{sklearn_row['test_accuracy'].values[0]:.4f}",
            f"{our_row['test_f1'].values[0]:.4f}",
            f"{sklearn_row['test_f1'].values[0]:.4f}",
            f"{our_row['training_time'].values[0]:.3f}s",
            f"{sklearn_row['training_time'].values[0]:.3f}s"
        ])

table = ax.table(cellText=summary_data,
                colLabels=['Feature Set', 'Our Acc', 'Sklearn Acc', 'Our F1', 'Sklearn F1', 
                          'Our Time', 'Sklearn Time'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Summary Comparison: Our Implementation vs Scikit-learn', 
         fontsize=16, fontweight='bold', pad=20)
plt.savefig(os.path.join(PLOTS_FOLDER, 'summary_comparison_table.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_FOLDER}")
print(f"Plots saved to: {PLOTS_FOLDER}")
print("\nGenerated files:")
print("  - comparison_results.csv")
print("  - plots/performance_comparison_*.png")
print("  - plots/runtime_comparison.png")
print("  - plots/performance_difference_heatmap.png")
print("  - plots/summary_comparison_table.png")

