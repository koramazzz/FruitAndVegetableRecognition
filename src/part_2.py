import json
import time

import umap  # requires: pip install umap-learn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from src.classification import build_model_configs

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results"

X_PATH = DATA_DIR / "X_final.npy"
Y_PATH = DATA_DIR / "y_final.npy"

RAW_DIR = BASE_DIR / "dataset" / "raw"
METADATA_PATH = RAW_DIR / "metadata.csv"
DESCRIPTION_PATH = RAW_DIR / "description.csv"

RANDOM_STATE = 42
CV_SPLITS = 3
TEST_SIZE = 0.2


def load_data():
    """Load numpy arrays and encode labels."""
    X = np.load(X_PATH)
    # y_final.npy stores string labels as an object array; need allow_pickle=True
    y_raw = np.load(Y_PATH, allow_pickle=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    print(f"Data: X {X.shape}, y {y.shape}, num classes: {len(class_names)}")
    return X, y, class_names

def run_benchmark(X_train, X_test, y_train, y_test):

    configs = build_model_configs()
    results = []

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== {name} ===")
        search = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cfg["cv"],
            n_jobs=-1,
            scoring="accuracy",
            refit=True,
            verbose=0,
        )

        print(f"{X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        start = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        best_est = search.best_estimator_
        y_pred = best_est.predict(X_test)
        print(y_test.shape, y_pred.shape)
        test_acc = accuracy_score(y_test, y_pred)

        result = {
            "model": name,
            "best_params": json.dumps(search.best_params_),
            "cv_mean_accuracy": search.best_score_,
            "test_accuracy": test_acc,
            "fit_time_sec": fit_time,
        }
        results.append(result)

        print(f"Best params: {search.best_params_}")
        print(f"CV mean accuracy: {search.best_score_:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Fit time (s): {fit_time:.2f}")

    # Persist results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values(by="cv_mean_accuracy", ascending=False)
    out_csv = RESULTS_DIR / "classification_benchmark_PCA.csv"
    results_df.to_csv(out_csv, index=False)

    # Nicely formatted table for console
    summary_df = results_df.copy()
    summary_df["cv_mean_accuracy"] = summary_df["cv_mean_accuracy"].map(lambda x: f"{x:.4f}")
    summary_df["test_accuracy"] = summary_df["test_accuracy"].map(lambda x: f"{x:.4f}")
    summary_df["fit_time_sec"] = summary_df["fit_time_sec"].map(lambda x: f"{x:.2f}")
    summary_df["best_params"] = summary_df["best_params"].apply(
        lambda s: s if len(s) <= 80 else s[:77] + "..."
    )

    print("\n--- Summary (table) ---")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {out_csv}")

    # Visual summary (accuracy and fit time)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Test accuracy bar
    axes[0].bar(results_df["model"], results_df["test_accuracy"], color="#4c72b0")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy by Model")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Fit time bar (log scale for readability)
    axes[1].bar(results_df["model"], results_df["fit_time_sec"], color="#dd8452")
    axes[1].set_ylabel("Fit Time (s)")
    axes[1].set_title("Training Time by Model")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "classification_benchmark_PCA.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

def task_2_1_pca_analysis(X_train, y_train, X_test, y_test, make_rec=False):
    X_scaled = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    print("--- Task 2.1: PCA Analysis ---")

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)

    mse_errors = []
    n_features = X_scaled.shape[1]
    if make_rec:
        for k in range(1, n_features + 1):
            if k%10==0:print(f"opt dmn loop at k={k}")
            pca_k = PCA(n_components=k)
            X_reduced = pca_k.fit_transform(X_scaled)
            X_reconstructed = pca_k.inverse_transform(X_reduced)
            mse = np.mean((X_scaled - X_reconstructed) ** 2)
            mse_errors.append(mse)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance', color=color)
    ax1.plot(range(1, n_features + 1), cum_var, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')

    if make_rec:
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Reconstruction Error (MSE)', color=color)
        ax2.plot(range(1, n_features + 1), mse_errors, color=color, marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title('PCA: Intrinsic Dimensionality Analysis')
    fig.tight_layout()
    plt.show()

    # Determine intrinsic dimension (e.g., k for 95% variance)
    k_optimal = np.argmax(cum_var >= 0.95) + 1
    print(f"Optimal k (95% variance): {k_optimal}")

    # B. Repeat Classification (Task 1, Q1) with lower-dimensional features
    print(f"\nTraining classifiers on reduced data (k={k_optimal})...")

    pca_opt = PCA(n_components=k_optimal)
    X_train_pca = pca_opt.fit_transform(X_train)
    X_test_pca = pca_opt.transform(X_test)

    run_benchmark(X_train_pca, X_test_pca, y_train, y_test)


def task_2_2_clustering(X_scaled, y_true, knn_ninit=10):
    print("\n--- Task 2.2: Clustering Analysis (Baseline vs. PCA vs. UMAP vs. t-SNE) ---")

    n_classes = len(np.unique(y_true))

    # =======================================================
    # 1. BASELINE: Clustering on Original Scaled Data
    # =======================================================
    print(f"[1/4] Applying K-Means on Original Data (k={n_classes})...")
    kmeans_orig = KMeans(n_clusters=n_classes, random_state=42, n_init=knn_ninit)
    labels_orig = kmeans_orig.fit_predict(X_scaled)

    sil_orig = silhouette_score(X_scaled, labels_orig)
    ari_orig = adjusted_rand_score(y_true, labels_orig)
    print(f"      Original Data -> Silhouette: {sil_orig:.4f} | ARI: {ari_orig:.4f}")

    # =======================================================
    # 2. PCA: Dimensionality Reduction & Clustering
    # =======================================================
    print("[2/4] Running PCA reduction...")
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"      Applying K-Means on PCA Features...")
    kmeans_pca = KMeans(n_clusters=n_classes, random_state=42, n_init=knn_ninit)
    labels_pca = kmeans_pca.fit_predict(X_pca)

    sil_pca = silhouette_score(X_pca, labels_pca)
    ari_pca = adjusted_rand_score(y_true, labels_pca)
    print(f"      PCA Features -> Silhouette: {sil_pca:.4f} | ARI: {ari_pca:.4f}")

    # =======================================================
    # 3. UMAP: Dimensionality Reduction & Clustering
    # =======================================================
    print("[3/4] Running UMAP reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    print(f"      Applying K-Means on UMAP Features...")
    kmeans_umap = KMeans(n_clusters=n_classes, random_state=42, n_init=knn_ninit)
    labels_umap = kmeans_umap.fit_predict(X_umap)

    sil_umap = silhouette_score(X_umap, labels_umap)
    ari_umap = adjusted_rand_score(y_true, labels_umap)
    print(f"      UMAP Features -> Silhouette: {sil_umap:.4f} | ARI: {ari_umap:.4f}")

    # =======================================================
    # 4. t-SNE: Dimensionality Reduction & Clustering
    # =======================================================
    print("[4/4] Running t-SNE reduction...")
    # perplexity=30 is standard, but you can tune it (5-50)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    print(f"      Applying K-Means on t-SNE Features...")
    kmeans_tsne = KMeans(n_clusters=n_classes, random_state=42, n_init=knn_ninit)
    labels_tsne = kmeans_tsne.fit_predict(X_tsne)

    sil_tsne = silhouette_score(X_tsne, labels_tsne)
    ari_tsne = adjusted_rand_score(y_true, labels_tsne)
    print(f"      t-SNE Features -> Silhouette: {sil_tsne:.4f} | ARI: {ari_tsne:.4f}")

    # =======================================================
    # 5. Visualization (4x2 Grid)
    # =======================================================
    print("Generating plots...")

    # LDA is used ONLY to visualize the "Original" high-dim results
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y_true)

    fig, axes = plt.subplots(4, 2, figsize=(16, 26))  # Increased height for 4 rows

    # --- ROW 1: Baseline (Original Data) ---
    axes[0, 0].scatter(X_lda[:, 0], X_lda[:, 1], c=y_true, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[0, 0].set_title('Ground Truth (LDA Projection)')

    axes[0, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=labels_orig, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[0, 1].set_title(f'K-Means on ORIGINAL Data\nARI: {ari_orig:.2f} | Sil: {sil_orig:.2f}')

    # --- ROW 2: PCA ---
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[1, 0].set_title('Ground Truth (PCA Projection)')
    axes[1, 0].set_xlabel('PCA 1')
    axes[1, 0].set_ylabel('PCA 2')

    axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[1, 1].set_title(f'K-Means on PCA Features\nARI: {ari_pca:.2f} | Sil: {sil_pca:.2f}')
    axes[1, 1].set_xlabel('PCA 1')
    axes[1, 1].set_ylabel('PCA 2')

    # --- ROW 3: UMAP ---
    axes[2, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=y_true, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[2, 0].set_title('Ground Truth (UMAP Projection)')
    axes[2, 0].set_xlabel('UMAP 1')
    axes[2, 0].set_ylabel('UMAP 2')

    axes[2, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=labels_umap, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[2, 1].set_title(f'K-Means on UMAP Features\nARI: {ari_umap:.2f} | Sil: {sil_umap:.2f}')
    axes[2, 1].set_xlabel('UMAP 1')
    axes[2, 1].set_ylabel('UMAP 2')

    # --- ROW 4: t-SNE ---
    axes[3, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[3, 0].set_title('Ground Truth (t-SNE Projection)')
    axes[3, 0].set_xlabel('t-SNE 1')
    axes[3, 0].set_ylabel('t-SNE 2')

    axes[3, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_tsne, cmap='nipy_spectral', s=15, alpha=0.7)
    axes[3, 1].set_title(f'K-Means on t-SNE Features\nARI: {ari_tsne:.2f} | Sil: {sil_tsne:.2f}')
    axes[3, 1].set_xlabel('t-SNE 1')
    axes[3, 1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.show()

    return {
        "kmeans_orig": kmeans_orig, "labels_orig": labels_orig,
        "kmeans_pca": kmeans_pca, "labels_pca": labels_pca,
        "kmeans_umap": kmeans_umap, "labels_umap": labels_umap,
        "kmeans_tsne": kmeans_tsne, "labels_tsne": labels_tsne,
        "X_tsne": X_tsne,
        "X_pca": X_pca
    }


def task_2_3_outlier_detection(X_pca, kmeans_pca, X_scaled, y_true, percentile=95):
    """
    Detects outliers based on PCA + K-Means and visualizes them
    on both PCA and LDA 2D planes.
    """
    df_meta = pd.read_csv(METADATA_PATH)
    df_desc = pd.read_csv(DESCRIPTION_PATH)
    df = pd.merge(df_meta, df_desc, on='ID')

    print("\n--- Task 2.3: Outlier Detection Comparison (PCA Detection + LDA Vis) ---")

    # Helper function to detect outliers for a single method
    def detect_outliers(X, model, name):
        # Calculate distances to nearest centroid
        all_dists = model.transform(X)
        min_dists = np.min(all_dists, axis=1)

        # Determine threshold
        threshold = np.percentile(min_dists, percentile)
        mask = min_dists > threshold
        indices = np.where(mask)[0]

        print(f"[{name}] Threshold: {threshold:.4f} | Outliers detected: {len(indices)}")
        return mask, indices, model.cluster_centers_, min_dists

    # 1. Run detection (Unsupervised - using PCA)
    mask_pca, idx_pca, centers_pca, min_dists_pca = detect_outliers(X_pca, kmeans_pca, "PCA")

    # Save results
    df['outlier_score'] = min_dists_pca
    outlier_csv = df[['ID', 'label', 'outlier_score']][mask_pca]
    outlier_csv_path = RESULTS_DIR / "detected_outliers_unsupervised.csv"
    outlier_csv.to_csv(outlier_csv_path, index=False)
    print(f"Saved outlier list to {outlier_csv_path}")

    # 2. Compute LDA Projection (Supervised - for visualization only)
    print("Computing LDA projection for visualization...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y_true)

    # 3. Visualization (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Outliers in PCA Space (Detection Space) ---
    axes[0].scatter(X_pca[~mask_pca, 0], X_pca[~mask_pca, 1],
                    c='gray', alpha=0.3, s=20, label='Normal')
    axes[0].scatter(X_pca[mask_pca, 0], X_pca[mask_pca, 1],
                    c='orange', alpha=0.9, s=50, edgecolors='k', label='Outliers')
    axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                    c='blue', s=200, marker='X', edgecolors='white', label='Centroids')

    axes[0].set_title(f"Outliers in PCA Space\n(Threshold: {percentile}th percentile)")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    # --- Plot 2: Same Outliers in LDA Space (Supervised Space) ---
    axes[1].scatter(X_lda[~mask_pca, 0], X_lda[~mask_pca, 1],
                    c='gray', alpha=0.3, s=20, label='Normal')
    axes[1].scatter(X_lda[mask_pca, 0], X_lda[mask_pca, 1],
                    c='red', alpha=0.9, s=50, edgecolors='k', label='Detected Outliers')

    axes[1].set_title(f"The Same Outliers Projected in LDA Space\n(Visual Inspection)")
    axes[1].set_xlabel("LDA 1")
    axes[1].set_ylabel("LDA 2")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return {
        "indices_pca": idx_pca
    }


if __name__ == "__main__":
    X, y, class_names = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split for classification tasks
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        test_size=TEST_SIZE,
                                                        stratify=y,
                                                        random_state=RANDOM_STATE, )

    # Run the analysis
    task_2_1_pca_analysis(X_train, y_train, X_test, y_test, make_rec=False)

    # Run Task 2.2
    clustering_results = task_2_2_clustering(X_scaled, y, knn_ninit=100)

    # Run Task 2.3 (Updated with X_scaled and y for LDA)
    outlier_results = task_2_3_outlier_detection(
        X_pca=clustering_results["X_pca"],
        kmeans_pca=clustering_results["kmeans_pca"],
        X_scaled=X_scaled,  # <--- Added
        y_true=y,  # <--- Added
        percentile=95
    )

