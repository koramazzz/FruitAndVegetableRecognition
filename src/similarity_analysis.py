import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
INPUT_FOLDER = '../dataset/processed'
X_PATH = os.path.join(INPUT_FOLDER, 'X_final.npy')
Y_PATH = os.path.join(INPUT_FOLDER, 'y_final.npy')

# Define the slice indices based on dimensions of the features
DIMS_META = 18
DIMS_TEXT = 280  # Reduced from 384 using PCA
DIMS_IMG = 201   # Reduced from 285 using PCA

def analyze_modality(X, y, modality_name):
    print(f"\n{'='*20} ANALYSIS: {modality_name} {'='*20}")
    classes = np.unique(y)
    
    # 1. Calculate Centroids
    centroids = {}
    for cls in classes:
        centroids[cls] = np.mean(X[y == cls], axis=0)

    # 2. Intra-Class and Inter-Class Similarity
    print(f"\n--- Average Cosine Similarity Matrix (0.0 to 1.0) ---")
    print(f"{'':10} | " + " | ".join([f"{c:>8}" for c in classes]))
    
    for cls_a in classes:
        row_str = f"{cls_a:10} | "
        for cls_b in classes:
            # Get samples for these classes
            samples_a = X[y == cls_a]
            samples_b = X[y == cls_b]
            
            # Compute pairwise similarity between group A and group B
            sim_matrix = cosine_similarity(samples_a, samples_b)
            avg_sim = np.mean(sim_matrix)
            
            row_str += f"{avg_sim:>8.4f} | "
        print(row_str)
        
    # 3. Outlier Detection (Distance to Centroid)
    print(f"\n--- Outlier Detection (Threshold: Mean Distance + 2 * StdDev) ---")
    total_outliers = 0
    
    for cls in classes:
        # Get samples for this class
        class_indices = np.where(y == cls)[0]
        samples = X[class_indices]
        centroid = centroids[cls]
        
        # Calculate Euclidean distance of each sample to the centroid
        distances = np.linalg.norm(samples - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + (2 * std_dist)
        
        # Identify outliers
        outlier_mask = distances > threshold
        num_outliers = np.sum(outlier_mask)
        total_outliers += num_outliers
        
        if num_outliers > 0:
            print(f"Class {cls}: Found {num_outliers} outliers (Threshold: {threshold:.4f})")
        else:
            print(f"Class {cls}: No significant outliers found.")

    return total_outliers

# --- MAIN EXECUTION ---

# 1. Load Data
if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    print("Error: Processed data not found. Run raw_to_vector.py first.")
    exit()

X_final = np.load(X_PATH)
y_final = np.load(Y_PATH, allow_pickle=True)

print(f"Loaded Data: X shape {X_final.shape}, y shape {y_final.shape}")

# 2. Slice the Data
# Indices
end_meta = DIMS_META
end_text = DIMS_META + DIMS_TEXT

X_meta = X_final[:, :end_meta]
X_text = X_final[:, end_meta:end_text]
X_image = X_final[:, end_text:]

# 3. Analyze Each Modality
print(f"Dimensions: Meta={X_meta.shape[1]}, Text={X_text.shape[1]}, Image={X_image.shape[1]}")

analyze_modality(X_meta, y_final, "METADATA (Cat/Num)")
analyze_modality(X_text, y_final, "TEXT DESCRIPTIONS")
analyze_modality(X_image, y_final, "IMAGES")
analyze_modality(X_final, y_final, "FUSED (ALL)")