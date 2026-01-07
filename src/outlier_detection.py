"""
Task 1.4: Outlier Detection Framework using SVM Constraints (Slack Variables).

This script:
1. Trains a Soft-Margin Linear SVM on the dataset.
2. Calculates the 'Slack Variable' (Xi) for every data point.
   - Xi > 1.0 means the point is misclassified or heavily violating the margin.
   - High Xi = High "Outlier Score".
3. Saves outlier results to CSV file.
4. Visualizes the Top 5 Outliers (images with the highest Xi scores).

Updates:
- Increased max_iter to prevent ConvergenceWarning.
- Improved precision in plot titles (showing 5 decimal places).
- Added CSV export for outlier results.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RAW_DIR = BASE_DIR / "dataset" / "raw"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_ORIGINAL_DIR = BASE_DIR / "dataset" / "images" / "original"
IMAGES_GENERATED_DIR = BASE_DIR / "dataset" / "images" / "generated"

X_PATH = DATA_DIR / "X_final.npy"
Y_PATH = DATA_DIR / "y_final.npy"
METADATA_PATH = RAW_DIR / "metadata.csv"
DESCRIPTION_PATH = RAW_DIR / "description.csv"

def find_image_path(sample_id, label):
    """
    Helper to locate the image file on disk (matches logic from raw_to_vector.py).
    """
    label_lower = label.lower()
    
    # Check original folder
    paths_to_check = [
        IMAGES_ORIGINAL_DIR / label_lower / f"{sample_id}_result.jpg",
        IMAGES_ORIGINAL_DIR / label_lower / f"{sample_id}.jpg",
    ]
    for p in paths_to_check:
        if p.exists(): return str(p)
    
    # Check generated folder
    gen_path = IMAGES_GENERATED_DIR / f"{label_lower}_gen" / f"{sample_id}.jpg"
    if gen_path.exists(): return str(gen_path)
    return None

def main():
    print("--- 1. Loading Data ---")
    X = np.load(X_PATH)
    y_raw = np.load(Y_PATH, allow_pickle=True)
    
    # Load metadata to find filenames (We need to reconstruct the dataframe to map index -> ID)
    # Note: We assume the order in X matches the sorted/processed order from raw_to_vector.py
    # Re-loading and merging exactly as Task 0 did to ensure alignment
    df_meta = pd.read_csv(METADATA_PATH)
    df_desc = pd.read_csv(DESCRIPTION_PATH)
    df = pd.merge(df_meta, df_desc, on='ID')
    
    print(f"Loaded {X.shape[0]} samples.")

    print("\n--- 2. Training SVM & Calculating Outlier Scores ---")
    # We use LinearSVC with 'hinge' loss to strictly enforce the standard SVM formulation
    # min ||w||^2 + C * sum(xi)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    
    # Train the model
    # We use C=1.0 consistent with our benchmarks
    # INCREASED max_iter to 50,000 to fix convergence warning
    print("Training LinearSVC (this might take a few seconds)...")
    clf = LinearSVC(loss='hinge', C=1.0, random_state=42, max_iter=50000)
    clf.fit(X_scaled, y_enc)
    
    # --- CALCULATE SLACK VARIABLES (Xi) ---
    # The decision_function returns the signed distance to the hyperplane for each class.
    decision_scores = clf.decision_function(X_scaled)  # (n_samples, n_classes)

    
    # For each sample, we only care about the score for its TRUE class.
    # Logic: For One-vs-Rest, we want y_true * score >= 1.
    # Slack (Xi) = max(0, 1 - y_true * score)
    
    outlier_scores = []
    
    for i in range(len(X)):
        true_class_idx = y_enc[i]
        
        # Get the score the model assigned to the TRUE class of this image
        # Note: If n_classes=2, decision_function returns 1D array. 
        # Assuming multi-class (>2) here based on dataset.
        if decision_scores.ndim > 1:
            score_for_true_class = decision_scores[i, true_class_idx]
        else:
            # Binary case handling (if needed)
            score = decision_scores[i]
            score_for_true_class = score if true_class_idx == 1 else -score

        # Calculate Slack: How much did it violate the margin?
        # A normal point has score > 1, so (1 - score) is negative -> max(0, neg) = 0 slack.
        # An outlier has score < 1 (or negative), so (1 - score) is positive.
        xi = max(0, 1 - score_for_true_class)
        outlier_scores.append(xi)

    df['outlier_score'] = outlier_scores
    
    print("\n--- 3. Identifying Top Outliers ---")
    # Sort by outlier score descending (Highest slack = Worst Outlier)
    df_sorted = df.sort_values(by='outlier_score', ascending=False)
    top_outliers = df_sorted.head(5)
    
    print("Top 5 Detected Outliers:")
    print(top_outliers[['ID', 'label', 'outlier_score']])
    
    # Check if we have any "Real" outliers (Slack > 1.0)
    real_outliers = df[df['outlier_score'] > 1.0]
    print(f"\n[Analysis] Number of 'Major' Outliers (Slack > 1.0): {len(real_outliers)}")
    if len(real_outliers) == 0:
        print(" -> Interpretation: The dataset is PERFECTLY separable. Even the worst points are safe!")
    
    # Save outliers to CSV
    print("\n--- 4. Saving Results to CSV ---")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select relevant columns for CSV output
    outlier_csv = df_sorted[['ID', 'label', 'outlier_score']].copy()
    outlier_csv_path = RESULTS_DIR / "detected_outliers.csv"
    outlier_csv.to_csv(outlier_csv_path, index=False)
    print(f"Saved outlier results to: {outlier_csv_path}")
    print(f"Total samples analyzed: {len(outlier_csv)}")
    
    print("\n--- 5. Visualizing (Plotting) ---")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Top 5 Outliers (Points Closest to the Margin)', fontsize=16)
    
    for idx, (_, row) in enumerate(top_outliers.iterrows()):
        ax = axes[idx]
        sample_id = row['ID']
        label = row['label']
        score = row['outlier_score']
        
        # Load Image
        img_path = find_image_path(sample_id, label)
        
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Corrupt File", ha='center')
        else:
            ax.text(0.5, 0.5, "Not Found", ha='center')
            
        # Showing 5 decimal places
        ax.set_title(f"{label}\nID: {sample_id}\nSlack: {score:.5f}", color='red', fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()