"""Soft-margin Linear SVM from Scratch.

Implements a soft-margin linear SVM using quadratic programming (cvxopt).
- Finds support vectors
- Finds data points farthest from the hyperplane in each category
- Supports multiclass via One-vs-Rest (OvR)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Suppress cvxopt output
cvxopt.solvers.options["show_progress"] = False

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results"

RAW_DIR = BASE_DIR / "dataset" / "raw"
IMAGES_ORIGINAL_DIR = BASE_DIR / "dataset" / "images" / "original"
IMAGES_GENERATED_DIR = BASE_DIR / "dataset" / "images" / "generated"

X_PATH = DATA_DIR / "X_final.npy"
Y_PATH = DATA_DIR / "y_final.npy"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# Helper Functions
# ============================================================================
def find_image_path(sample_id, label):
    """Locate image file on disk."""
    label_lower = label.lower()
    paths = [
        IMAGES_ORIGINAL_DIR / label_lower / f"{sample_id}_result.jpg",
        IMAGES_ORIGINAL_DIR / label_lower / f"{sample_id}.jpg",
        IMAGES_GENERATED_DIR / f"{label_lower}_gen" / f"{sample_id}.jpg"
    ]
    for p in paths:
        if p.exists(): return str(p)
    return None

def visualize_farthest_points(farthest_points_dict):
    """Show the specific farthest points in a window."""
    # We want to show 4 specific images as requested (Banana +/- and Cucumber +/-)
    # Or generically, the first few. Let's do a generic grid.
    
    # Collect points to plot
    points_to_plot = []
    for class_name, data in farthest_points_dict.items():
        if "positive_class" in data:
            d = data["positive_class"]
            points_to_plot.append((class_name, "+1 (Typical)", d))
        if "negative_class" in data:
            d = data["negative_class"]
            points_to_plot.append((class_name, "-1 (Outlier)", d))
            
    # Limit to first 4 for the specific user request, or show more if you like
    # User asked for "Banana Pos/Neg" and "Cucumber Pos/Neg" specifically
    subset = points_to_plot[:4] 

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Farthest Points from Hyperplane (Top 4)', fontsize=16)
    
    for ax, (cls_name, side, info) in zip(axes, subset):
        img_path = find_image_path(info['id'], info['original_label'])
        if img_path:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"Classifier: {cls_name}\nSide: {side}\nTrue: {info['original_label']}\nID: {info['id']}")
        else:
            ax.text(0.5, 0.5, "Image Not Found")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_closest_pairs(df_distances):
    """Show top 2 closest pairs (4 images) from different classes."""
    # Filter for different classes
    diff_class = df_distances[df_distances['same_class'] == False].head(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Closest Support Vectors (Different Classes)', fontsize=16)
    
    rows = diff_class.to_dict('records')
    
    for i, row in enumerate(rows):
        # Pair 1: Image A
        ax1 = axes[i, 0]
        path1 = find_image_path(row['id1'], row['class1'])
        if path1:
            img = cv2.imread(path1)
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Pair {i+1}-A\n{row['class1']} ({row['id1']})")
        ax1.axis('off')

        # Pair 1: Image B
        ax2 = axes[i, 1]
        path2 = find_image_path(row['id2'], row['class2'])
        if path2:
            img = cv2.imread(path2)
            ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Pair {i+1}-B\n{row['class2']} ({row['id2']})\nDist: {row['distance']:.2f}")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Soft-Margin Linear SVM (Binary) - From Scratch
# ============================================================================
class SoftMarginLinearSVM:
    """
    Soft-margin linear SVM trained via Quadratic Programming.
    
    Dual formulation:
        max_α  Σ α_i - (1/2) Σ_{i,j} α_i α_j y_i y_j (x_i · x_j)
        s.t.   0 ≤ α_i ≤ C   (box constraints)
               Σ α_i y_i = 0  (equality constraint)
    
    The weight vector: w = Σ α_i y_i x_i
    The bias: b is computed from support vectors where 0 < α < C
    """
    
    def __init__(self, C: float = 1.0, tol: float = 1e-5):
        """
        Args:
            C: Regularization parameter (penalty for misclassification)
            tol: Tolerance for identifying support vectors (α > tol)
        """
        self.C = C
        self.tol = tol
        self.alphas: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_vector_indices_: Optional[np.ndarray] = None
        self.support_vector_labels_: Optional[np.ndarray] = None
        self.support_vector_alphas_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftMarginLinearSVM":
        """
        Train the SVM using quadratic programming.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, must be +1 or -1, shape (n_samples,)
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        self._X_train = X.copy()
        self._y_train = y.copy()
        
        # Ensure y is float for matrix operations
        y = y.astype(np.float64)
        
        # Compute the Gram matrix: K_ij = x_i · x_j
        K = X @ X.T
        
        # Build the quadratic term: Q_ij = y_i y_j (x_i · x_j)
        # For cvxopt: minimize (1/2) x^T P x + q^T x
        # Our dual (maximization) becomes minimization with sign flip
        Q = np.outer(y, y) * K
        
        # Convert to cvxopt matrices
        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(-np.ones(n_samples))  # -1 vector (we minimize, so negate)
        
        # Inequality constraints: Gx <= h
        # We need: α >= 0  =>  -α <= 0
        #          α <= C  =>   α <= C
        G_upper = np.eye(n_samples)      # α <= C
        G_lower = -np.eye(n_samples)     # -α <= 0  (i.e., α >= 0)
        G = cvxopt.matrix(np.vstack([G_upper, G_lower]))
        
        h_upper = np.full(n_samples, self.C)
        h_lower = np.zeros(n_samples)
        h = cvxopt.matrix(np.hstack([h_upper, h_lower]))
        
        # Equality constraint: Σ α_i y_i = 0
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))
        
        # Solve QP
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        if solution["status"] != "optimal":
            print(f"Warning: QP solver status = {solution['status']}")
        
        # Extract alphas
        alphas = np.array(solution["x"]).flatten()
        self.alphas = alphas
        
        # Identify support vectors (α > tol)
        sv_mask = alphas > self.tol
        self.support_vector_indices_ = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.support_vector_alphas_ = alphas[sv_mask]
        
        # Compute weight vector: w = Σ α_i y_i x_i
        self.w = np.sum((alphas * y)[:, np.newaxis] * X, axis=0)
        
        # Compute bias using support vectors on the margin (0 < α < C)
        # For these: y_i (w · x_i + b) = 1  =>  b = y_i - w · x_i
        margin_sv_mask = (alphas > self.tol) & (alphas < self.C - self.tol)
        if np.any(margin_sv_mask):
            margin_indices = np.where(margin_sv_mask)[0]
            b_values = y[margin_indices] - X[margin_indices] @ self.w
            self.b = np.mean(b_values)
        else:
            # Fallback: use all support vectors
            sv_indices = self.support_vector_indices_
            if len(sv_indices) > 0:
                b_values = y[sv_indices] - X[sv_indices] @ self.w
                self.b = np.mean(b_values)
            else:
                self.b = 0.0
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the signed distance to the hyperplane."""
        return X @ self.w + self.b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (+1 or -1)."""
        return np.sign(self.decision_function(X))
    
    def get_distances_to_hyperplane(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the signed distance of each point to the hyperplane.
        
        Distance = (w · x + b) / ||w||
        Positive distance = on the +1 side
        Negative distance = on the -1 side
        """
        norm_w = np.linalg.norm(self.w)
        if norm_w < 1e-10:
            return np.zeros(X.shape[0])
        return self.decision_function(X) / norm_w


# ============================================================================
# One-vs-Rest Multiclass Wrapper
# ============================================================================
class MulticlassSVM:
    """One-vs-Rest multiclass SVM using binary SoftMarginLinearSVM."""
    
    def __init__(self, C: float = 1.0, tol: float = 1e-5):
        self.C = C
        self.tol = tol
        self.classifiers_: Dict[int, SoftMarginLinearSVM] = {}
        self.classes_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassSVM":
        """Train one binary SVM per class (One-vs-Rest)."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        print(f"Training {n_classes} One-vs-Rest binary SVMs...")
        
        for i, cls in enumerate(self.classes_):
            print(f"  Training classifier for class {cls} ({i+1}/{n_classes})...")
            # Create binary labels: +1 for this class, -1 for others
            y_binary = np.where(y == cls, 1, -1).astype(np.float64)
            
            clf = SoftMarginLinearSVM(C=self.C, tol=self.tol)
            clf.fit(X, y_binary)
            self.classifiers_[cls] = clf
            
            n_sv = len(clf.support_vector_indices_)
            print(f"    Found {n_sv} support vectors")
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision values for each class."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decisions = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(self.classes_):
            decisions[:, i] = self.classifiers_[cls].decision_function(X)
        
        return decisions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class with highest decision value."""
        decisions = self.decision_function(X)
        return self.classes_[np.argmax(decisions, axis=1)]
    
    def get_all_support_vectors_info(self, X: np.ndarray, y: np.ndarray, 
                                      class_names: np.ndarray) -> Dict:
        """
        Get comprehensive information about support vectors across all classifiers.
        
        Returns:
            Dictionary with support vector information per class
        """
        info = {}
        
        for cls in self.classes_:
            clf = self.classifiers_[cls]
            class_name = class_names[cls]
            
            # Get indices of support vectors (global indices in X)
            sv_indices = clf.support_vector_indices_
            
            info[class_name] = {
                "class_id": cls,
                "n_support_vectors": len(sv_indices),
                "support_vector_indices": sv_indices,
                "support_vector_alphas": clf.support_vector_alphas_,
                "support_vector_labels": clf.support_vector_labels_,
                "w": clf.w,
                "b": clf.b,
            }
        
        return info


# ============================================================================
# Analysis Functions
# ============================================================================
def find_farthest_points(clf: SoftMarginLinearSVM, X: np.ndarray, y: np.ndarray,
                         y_original: np.ndarray, class_names: np.ndarray, 
                         ids: np.ndarray) -> Dict: # <--- Added ids arg
    distances = clf.get_distances_to_hyperplane(X)
    result = {}
    
    # Helper to package result
    def make_info(idx):
        return {
            "index": idx,
            "id": ids[idx], # <--- Store the string ID
            "distance": distances[idx],
            "original_label": class_names[y_original[idx]],
        }

    # Positive class
    pos_mask = y == 1
    if np.any(pos_mask):
        pos_indices = np.where(pos_mask)[0]
        pos_distances = distances[pos_mask]
        farthest_pos_idx = pos_indices[np.argmax(pos_distances)]
        result["positive_class"] = make_info(farthest_pos_idx)
    
    # Negative class
    neg_mask = y == -1
    if np.any(neg_mask):
        neg_indices = np.where(neg_mask)[0]
        neg_distances = distances[neg_mask]
        farthest_neg_idx = neg_indices[np.argmin(neg_distances)]
        result["negative_class"] = make_info(farthest_neg_idx)
    
    return result

def analyze_support_vectors(multiclass_svm: MulticlassSVM, X: np.ndarray, 
                            y: np.ndarray, class_names: np.ndarray,
                            ids: np.ndarray) -> Dict: # <--- Added ids arg
    analysis = {
        "per_class": {},
        "global_support_vector_indices": set(),
        "farthest_points": {},
    }
    
    for cls in multiclass_svm.classes_:
        class_name = class_names[cls]
        clf = multiclass_svm.classifiers_[cls]
        y_binary = np.where(y == cls, 1, -1).astype(np.float64)
        
        sv_indices = clf.support_vector_indices_
        analysis["global_support_vector_indices"].update(sv_indices.tolist())
        
        sv_original_labels = y[sv_indices]
        sv_by_class = {}
        for orig_cls in multiclass_svm.classes_:
            orig_name = class_names[orig_cls]
            mask = sv_original_labels == orig_cls
            sv_by_class[orig_name] = sv_indices[mask].tolist()
        
        # Pass ids to farthest points finder
        farthest = find_farthest_points(clf, X, y_binary, y, class_names, ids)
        
        analysis["per_class"][class_name] = {
            "n_support_vectors": len(sv_indices),
            "support_vector_indices": sv_indices.tolist(),
            "support_vectors_by_original_class": sv_by_class,
            "alphas": clf.support_vector_alphas_.tolist(),
            "w_norm": float(np.linalg.norm(clf.w)),
            "b": float(clf.b),
        }
        analysis["farthest_points"][class_name] = farthest
    
    analysis["global_support_vector_indices"] = sorted(list(analysis["global_support_vector_indices"]))
    analysis["n_unique_support_vectors"] = len(analysis["global_support_vector_indices"])
    return analysis

def compute_sv_pairwise_distances(X: np.ndarray, sv_indices: List[int], 
                                   y: np.ndarray, class_names: np.ndarray,
                                   ids: np.ndarray) -> pd.DataFrame: # <--- Added ids arg
    sv_indices = np.array(sv_indices)
    n_sv = len(sv_indices)
    rows = []
    
    # Optimization: To save time, we can just do the top closest in the main loop or
    # keep this loop but add ID info.
    for i in range(n_sv):
        for j in range(i + 1, n_sv):
            idx1, idx2 = sv_indices[i], sv_indices[j]
            dist = np.linalg.norm(X[idx1] - X[idx2])
            class1 = class_names[y[idx1]]
            class2 = class_names[y[idx2]]
            
            # Optimization: Only store if they are different classes (for this specific task)
            # or keep all. Storing all is safer for general use.
            rows.append({
                "idx1": idx1, "idx2": idx2,
                "id1": ids[idx1], "id2": ids[idx2], # <--- Store IDs
                "distance": dist,
                "class1": class1, "class2": class2,
                "same_class": class1 == class2,
            })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("distance").reset_index(drop=True)
    return df

# ============================================================================
# Main
# ============================================================================
def load_data():
    """Load numpy arrays, encode labels, and get String IDs."""
    X = np.load(X_PATH)
    y_raw = np.load(Y_PATH, allow_pickle=True)
    
    # Load metadata to get IDs (e.g., banana_001)
    df_meta = pd.read_csv(RAW_DIR / "metadata.csv")
    # Ensure alignment: We assume X was generated in same order as metadata
    ids = df_meta['ID'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    
    print(f"Data: X {X.shape}, y {y.shape}")
    return X, y, class_names, label_encoder, ids

def main():
    print("=" * 70)
    print("Task 1.2: Soft-Margin Linear SVM from Scratch")
    print("=" * 70)
    
    # 1. Load data WITH IDs
    X, y, class_names, label_encoder, ids = load_data()
    
    # 2. Split everything (including IDs)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiclass SVM
    C = 1.0 # Regularization parameter
    print(f"\nTraining Soft-Margin SVM with C={C}...")
    svm = MulticlassSVM(C=C, tol=1e-5)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = svm.predict(X_train_scaled)
    y_pred_test = svm.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    # 3. Analyze with IDs
    print(f"\n{'='*70}\nSUPPORT VECTOR ANALYSIS\n{'='*70}")
    analysis = analyze_support_vectors(svm, X_train_scaled, y_train, class_names, ids_train)
    
    print(f"\nTotal unique support vectors: {analysis['n_unique_support_vectors']} "
          f"(out of {X_train.shape[0]} training samples)")
    
    print("\n--- Per-Class Binary Classifier Info ---")
    for class_name, info in analysis["per_class"].items():
        print(f"\n[{class_name} vs Rest]")
        print(f"  Support vectors: {info['n_support_vectors']}")
        print(f"  ||w||: {info['w_norm']:.4f}")
        print(f"  b: {info['b']:.4f}")
        print(f"  SVs by original class:")
        for orig_class, indices in info["support_vectors_by_original_class"].items():
            if len(indices) > 0:
                print(f"    {orig_class}: {len(indices)} SVs")

    # 4. Farthest Points Table & VISUALIZATION
    print(f"\n{'='*70}\nFARTHEST POINTS FROM HYPERPLANE\n{'='*70}")
    
    farthest_rows = []
    for class_name, farthest in analysis["farthest_points"].items():
        if "positive_class" in farthest:
            p = farthest["positive_class"]
            farthest_rows.append({
                "Classifier": f"{class_name} vs Rest", "Side": "Positive (+1)",
                "ID": p["id"], "Distance": p["distance"], "True_Label": p["original_label"]
            })
        if "negative_class" in farthest:
            n = farthest["negative_class"]
            farthest_rows.append({
                "Classifier": f"{class_name} vs Rest", "Side": "Negative (-1)",
                "ID": n["id"], "Distance": n["distance"], "True_Label": n["original_label"]
            })
    
    farthest_df = pd.DataFrame(farthest_rows)
    print(farthest_df.head(10).to_string(index=False)) # Show table in console
    farthest_df.to_csv(RESULTS_DIR / "svm_farthest_points.csv", index=False)
    
    print("\nVisualizing top 4 farthest points...")
    visualize_farthest_points(analysis["farthest_points"])

    # 5. Pairwise Distances & VISUALIZATION
    print(f"\n{'='*70}\nSUPPORT VECTOR PAIRWISE DISTANCES\n{'='*70}")
    
    # Pass ids_train here
    sv_distances = compute_sv_pairwise_distances(
        X_train_scaled, analysis["global_support_vector_indices"], y_train, class_names, ids_train
    )
    
    cross_class = sv_distances[~sv_distances["same_class"]]
    print(f"\nClosest support vector pairs from DIFFERENT classes:")
    print(cross_class[['id1', 'class1', 'id2', 'class2', 'distance']].head(10).to_string(index=False))
    
    sv_distances.to_csv(RESULTS_DIR / "svm_sv_pairwise_distances.csv", index=False)

    print("\nVisualizing closest pairs...")
    visualize_closest_pairs(sv_distances)
    
    print("\nDONE.")

if __name__ == "__main__":
    main()