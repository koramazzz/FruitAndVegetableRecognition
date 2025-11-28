# Task 2)a

import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import OneVsAllClassifier

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '../..')

INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'dataset/processed')
X_PATH = os.path.join(INPUT_FOLDER, 'X_final.npy')
Y_PATH = os.path.join(INPUT_FOLDER, 'y_final.npy')

OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'results')
LOSS_PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'loss_plots')

# Feature dimensions (after PCA reduction to 499 total)
DIMS_META = 18
DIMS_TEXT = 280  # Reduced from 384 using PCA
DIMS_IMG = 201  # Reduced from 285 using PCA

# Model hyperparameters
LEARNING_RATE = 0.1
MAX_ITER = 1000
TOLERANCE = 1e-6
REGULARIZATION = 'l2'
LAMBDA_REG = 0.1

# Fused model feature weights (only applied to fused model) 
FUSED_WEIGHT_METADATA = 0.5
FUSED_WEIGHT_TEXT = 0.5 
FUSED_WEIGHT_IMAGE = 1.0

# Train/Val split
VAL_SIZE = 500  # 500 samples for validation
TEST_SIZE = 500  # 500 samples for test
RANDOM_STATE = 42

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOSS_PLOTS_FOLDER, exist_ok=True)

print("=" * 70)
print("LOGISTIC REGRESSION TRAINING - TASK 2)a")
print("=" * 70)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n--- 1. LOADING DATA ---")
if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    print(f"Error: Processed data not found at {INPUT_FOLDER}")
    print("Please run raw_to_vector.py first to generate feature vectors.")
    exit(1)

X_final = np.load(X_PATH)
y_final = np.load(Y_PATH, allow_pickle=True)

print(f"Loaded data:")
print(f"  X shape: {X_final.shape}")
print(f"  y shape: {y_final.shape}")
print(f"  Classes: {np.unique(y_final)}")

# ==========================================
# 2. SPLIT FEATURE SETS
# ==========================================
print("\n--- 2. SPLITTING FEATURE SETS ---")

# Define slice indices
end_meta = DIMS_META
end_text = DIMS_META + DIMS_TEXT
end_img = DIMS_META + DIMS_TEXT + DIMS_IMG

# Extract individual feature sets
X_meta = X_final[:, :end_meta]
X_text = X_final[:, end_meta:end_text]
X_image = X_final[:, end_text:end_img]
X_fused = X_final  # All features combined

print(f"Feature dimensions:")
print(f"  Metadata: {X_meta.shape[1]} dims")
print(f"  Text:     {X_text.shape[1]} dims")
print(f"  Image:    {X_image.shape[1]} dims")
print(f"  Fused:    {X_fused.shape[1]} dims")

# ==========================================
# 3. TRAIN/VAL/TEST SPLIT
# ==========================================
print("\n--- 3. TRAIN/VAL/TEST SPLIT ---")

# Get indices for splitting (we'll use the same indices for all feature sets)
n_samples = len(X_final)
indices = np.arange(n_samples)

print(f"Total samples available: {n_samples}")
print(f"Target split: Test={TEST_SIZE}, Val={VAL_SIZE}, Train=remaining")

# Adjust split sizes based on available data
# If we don't have enough samples, use proportional splits
if n_samples < TEST_SIZE + VAL_SIZE:
    print(f"Warning: Only {n_samples} samples available (need {TEST_SIZE + VAL_SIZE} for test+val).")
    print(f"  Using proportional splits.")
    test_ratio = 0.14  # ~14% for test
    val_ratio = 0.14   # ~14% for validation (of remaining after test)
    
    # Calculate actual sizes
    test_size_actual = int(n_samples * test_ratio)
    remaining_after_test = n_samples - test_size_actual
    val_size_actual = int(remaining_after_test * val_ratio)
    
    print(f"  Using: Test={test_size_actual} ({test_ratio:.1%}), Val={val_size_actual} ({val_ratio:.1%}), Train=rest")
else:
    # We have enough samples, use fixed sizes
    test_size_actual = TEST_SIZE
    val_size_actual = VAL_SIZE
    print(f"  Using fixed sizes: Test={TEST_SIZE}, Val={VAL_SIZE}")

# First split: separate test set
indices_temp, indices_test, y_temp, y_test = train_test_split(
    indices, y_final,
    test_size=test_size_actual,
    random_state=RANDOM_STATE,
    stratify=y_final
)

# Second split: separate validation set from remaining data
indices_train, indices_val, y_train, y_val = train_test_split(
    indices_temp, y_temp,
    test_size=val_size_actual,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"\nFinal split sizes:")
print(f"  Training:   {len(indices_train)} samples ({len(indices_train)/n_samples:.1%})")
print(f"  Validation: {len(indices_val)} samples ({len(indices_val)/n_samples:.1%})")
print(f"  Test:       {len(indices_test)} samples ({len(indices_test)/n_samples:.1%})")

# Apply same splits to all feature sets using the same indices
X_meta_train = X_meta[indices_train]
X_meta_val = X_meta[indices_val]
X_meta_test = X_meta[indices_test]

X_text_train = X_text[indices_train]
X_text_val = X_text[indices_val]
X_text_test = X_text[indices_test]

X_image_train = X_image[indices_train]
X_image_val = X_image[indices_val]
X_image_test = X_image[indices_test]

print(f"\n--- Applying weights to fused features ---")
print(f"  Metadata weight: {FUSED_WEIGHT_METADATA}")
print(f"  Text weight:     {FUSED_WEIGHT_TEXT}")
print(f"  Image weight:    {FUSED_WEIGHT_IMAGE}")

# Weight each feature set
X_meta_weighted_train = X_meta_train * FUSED_WEIGHT_METADATA
X_meta_weighted_val = X_meta_val * FUSED_WEIGHT_METADATA
X_meta_weighted_test = X_meta_test * FUSED_WEIGHT_METADATA

X_text_weighted_train = X_text_train * FUSED_WEIGHT_TEXT
X_text_weighted_val = X_text_val * FUSED_WEIGHT_TEXT
X_text_weighted_test = X_text_test * FUSED_WEIGHT_TEXT

X_image_weighted_train = X_image_train * FUSED_WEIGHT_IMAGE
X_image_weighted_val = X_image_val * FUSED_WEIGHT_IMAGE
X_image_weighted_test = X_image_test * FUSED_WEIGHT_IMAGE

# Combine weighted features
X_fused_train = np.hstack([X_meta_weighted_train, X_text_weighted_train, X_image_weighted_train])
X_fused_val = np.hstack([X_meta_weighted_val, X_text_weighted_val, X_image_weighted_val])
X_fused_test = np.hstack([X_meta_weighted_test, X_text_weighted_test, X_image_weighted_test])

# ==========================================
# 4. TRAIN MODELS FOR EACH FEATURE SET
# ==========================================

feature_sets = {
    'images': {
        'X_train': X_image_train,
        'X_val': X_image_val,
        'X_test': X_image_test,
        'name': 'Images Only',
        'dims': X_image.shape[1]
    },
    'metadata': {
        'X_train': X_meta_train,
        'X_val': X_meta_val,
        'X_test': X_meta_test,
        'name': 'Categorical/Numerical Attributes',
        'dims': X_meta.shape[1]
    },
    'text': {
        'X_train': X_text_train,
        'X_val': X_text_val,
        'X_test': X_text_test,
        'name': 'Text Descriptions',
        'dims': X_text.shape[1]
    },
    'fused': {
        'X_train': X_fused_train,
        'X_val': X_fused_val,
        'X_test': X_fused_test,
        'name': f'Fused Features (All) [Meta:{FUSED_WEIGHT_METADATA}, Text:{FUSED_WEIGHT_TEXT}, Img:{FUSED_WEIGHT_IMAGE}]',
        'dims': X_fused_train.shape[1],
        'weights': {
            'metadata': FUSED_WEIGHT_METADATA,
            'text': FUSED_WEIGHT_TEXT,
            'image': FUSED_WEIGHT_IMAGE
        }
    }
}

trained_models = {}

for feature_type, data in feature_sets.items():
    print("\n" + "=" * 70)
    print(f"TRAINING MODEL: {data['name']}")
    print(f"Feature dimensions: {data['dims']}")
    print("=" * 70)
    
    # Create model
    model = OneVsAllClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        tolerance=TOLERANCE,
        regularization=REGULARIZATION,
        lambda_reg=LAMBDA_REG,
        verbose=True
    )
    
    # Train model and measure time
    import time
    start_time = time.time()
    model.fit(
        data['X_train'],
        y_train,
        data['X_val'],
        y_val
    )
    training_time = time.time() - start_time
    
    # Store trained model
    trained_models[feature_type] = {
        'model': model,
        'name': data['name'],
        'X_test': data['X_test'],
        'training_time': training_time,
        'config': {
            'learning_rate': LEARNING_RATE,
            'max_iter': MAX_ITER,
            'tolerance': TOLERANCE,
            'regularization': REGULARIZATION,
            'lambda_reg': LAMBDA_REG
        }
    }
    
    # Plot and save loss history
    print(f"\n--- Plotting loss history for {data['name']} ---")
    plt.figure(figsize=(12, 6))
    
    # Plot training loss for each class
    for class_label in model.classes:
        loss_history = model.get_loss_history(class_label)[class_label]
        plt.plot(loss_history, label=f'Training - {class_label}', alpha=0.7)
    
    # Plot validation loss for each class (if available)
    if len(model.val_loss_histories) > 0:
        for class_label in model.classes:
            if class_label in model.val_loss_histories:
                val_loss_history = model.val_loss_histories[class_label]
                plt.plot(val_loss_history, label=f'Validation - {class_label}', 
                        linestyle='--', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss History - {data["name"]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'loss_{feature_type}.png'
    plot_path = os.path.join(LOSS_PLOTS_FOLDER, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved loss plot to: {plot_path}")
    plt.close()
    
    # Check for overfitting
    print(f"\n--- Overfitting Check for {data['name']} ---")
    final_train_losses = []
    final_val_losses = []
    
    for class_label in model.classes:
        train_losses = model.get_loss_history(class_label)[class_label]
        final_train_losses.append(train_losses[-1])
        
        if class_label in model.val_loss_histories:
            val_losses = model.val_loss_histories[class_label]
            final_val_losses.append(val_losses[-1])
    
    avg_train_loss = np.mean(final_train_losses)
    avg_val_loss = np.mean(final_val_losses) if final_val_losses else None
    
    print(f"  Average final training loss: {avg_train_loss:.6f}")
    if avg_val_loss:
        print(f"  Average final validation loss: {avg_val_loss:.6f}")
        if avg_val_loss > avg_train_loss * 1.1:  # 10% threshold
            print(f"  WARNING: Possible overfitting detected!")
            print(f"     Consider increasing regularization (current lambda: {LAMBDA_REG})")
        else:
            print(f"  ✓ No significant overfitting detected")

print("\n" + "=" * 70)
print("TRAINING COMPLETED FOR ALL FEATURE SETS")
print("=" * 70)

# Save trained models using pickle
print("\n--- Saving trained models ---")
models_path = os.path.join(OUTPUT_FOLDER, 'trained_models.pkl')
with open(models_path, 'wb') as f:
    pickle.dump(trained_models, f)
print(f"  Models saved to: {models_path}")

# Also save train/test split indices and labels for evaluation
split_data = {
    'indices_train': indices_train,
    'indices_val': indices_val,
    'indices_test': indices_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'feature_sets': feature_sets,
    'fused_weights': {
        'metadata': FUSED_WEIGHT_METADATA,
        'text': FUSED_WEIGHT_TEXT,
        'image': FUSED_WEIGHT_IMAGE
    }
}
split_path = os.path.join(OUTPUT_FOLDER, 'split_data.pkl')
with open(split_path, 'wb') as f:
    pickle.dump(split_data, f)
print(f"  Split data saved to: {split_path}")

print(f"\nAll training completed!")
print(f"   Loss plots saved to: {LOSS_PLOTS_FOLDER}")
print(f"   Next step: Calculate metrics (accuracy, precision, recall, F1, AUC)")