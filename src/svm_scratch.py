"""Soft-margin Linear SVM from Scratch.

Implements a soft-margin linear SVM using quadratic programming (cvxopt).
- Finds support vectors
- Finds data points farthest from the hyperplane in each category
- Supports multiclass via One-vs-Rest (OvR)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cvxopt
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

X_PATH = DATA_DIR / "X_final.npy"
Y_PATH = DATA_DIR / "y_final.npy"

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
                         y_original: np.ndarray, class_names: np.ndarray) -> Dict:
    """
    Find data points farthest from the hyperplane in each category.
    
    For a binary classifier (One-vs-Rest), we find:
    - Farthest point in the positive class (+1 side, this class)
    - Farthest point in each negative class (-1 side, other classes)
    """
    distances = clf.get_distances_to_hyperplane(X)
    
    result = {}
    
    # Positive class (y == +1): find point with largest positive distance
    pos_mask = y == 1
    if np.any(pos_mask):
        pos_indices = np.where(pos_mask)[0]
        pos_distances = distances[pos_mask]
        farthest_pos_idx = pos_indices[np.argmax(pos_distances)]
        result["positive_class"] = {
            "index": farthest_pos_idx,
            "distance": distances[farthest_pos_idx],
            "original_label": class_names[y_original[farthest_pos_idx]],
        }
    
    # Negative class (y == -1): find point with most negative distance (farthest on -1 side)
    neg_mask = y == -1
    if np.any(neg_mask):
        neg_indices = np.where(neg_mask)[0]
        neg_distances = distances[neg_mask]
        farthest_neg_idx = neg_indices[np.argmin(neg_distances)]  # most negative
        result["negative_class"] = {
            "index": farthest_neg_idx,
            "distance": distances[farthest_neg_idx],
            "original_label": class_names[y_original[farthest_neg_idx]],
        }
    
    return result


def analyze_support_vectors(multiclass_svm: MulticlassSVM, X: np.ndarray, 
                            y: np.ndarray, class_names: np.ndarray) -> Dict:
    """
    Comprehensive analysis of support vectors.
    
    Returns dict with:
    - Support vector info per binary classifier
    - Farthest points from hyperplane per classifier
    - Global support vector indices (union across all classifiers)
    """
    analysis = {
        "per_class": {},
        "global_support_vector_indices": set(),
        "farthest_points": {},
    }
    
    for cls in multiclass_svm.classes_:
        class_name = class_names[cls]
        clf = multiclass_svm.classifiers_[cls]
        
        # Binary labels for this classifier
        y_binary = np.where(y == cls, 1, -1).astype(np.float64)
        
        # Support vector info
        sv_indices = clf.support_vector_indices_
        analysis["global_support_vector_indices"].update(sv_indices.tolist())
        
        # Categorize support vectors by their original class
        sv_original_labels = y[sv_indices]
        sv_by_class = {}
        for orig_cls in multiclass_svm.classes_:
            orig_name = class_names[orig_cls]
            mask = sv_original_labels == orig_cls
            sv_by_class[orig_name] = sv_indices[mask].tolist()
        
        # Farthest points
        farthest = find_farthest_points(clf, X, y_binary, y, class_names)
        
        analysis["per_class"][class_name] = {
            "n_support_vectors": len(sv_indices),
            "support_vector_indices": sv_indices.tolist(),
            "support_vectors_by_original_class": sv_by_class,
            "alphas": clf.support_vector_alphas_.tolist(),
            "w_norm": float(np.linalg.norm(clf.w)),
            "b": float(clf.b),
        }
        
        analysis["farthest_points"][class_name] = farthest
    
    analysis["global_support_vector_indices"] = sorted(
        list(analysis["global_support_vector_indices"])
    )
    analysis["n_unique_support_vectors"] = len(analysis["global_support_vector_indices"])
    
    return analysis


def compute_sv_pairwise_distances(X: np.ndarray, sv_indices: List[int], 
                                   y: np.ndarray, class_names: np.ndarray) -> pd.DataFrame:
    """
    Compute pairwise Euclidean distances between support vectors.
    
    Returns a DataFrame with columns: idx1, idx2, distance, class1, class2
    """
    sv_indices = np.array(sv_indices)
    n_sv = len(sv_indices)
    
    rows = []
    for i in range(n_sv):
        for j in range(i + 1, n_sv):
            idx1, idx2 = sv_indices[i], sv_indices[j]
            dist = np.linalg.norm(X[idx1] - X[idx2])
            class1 = class_names[y[idx1]]
            class2 = class_names[y[idx2]]
            rows.append({
                "idx1": idx1,
                "idx2": idx2,
                "distance": dist,
                "class1": class1,
                "class2": class2,
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
    """Load numpy arrays and encode labels."""
    X = np.load(X_PATH)
    y_raw = np.load(Y_PATH, allow_pickle=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    print(f"Data: X {X.shape}, y {y.shape}, num classes: {len(class_names)}")
    print(f"Classes: {list(class_names)}")
    return X, y, class_names, label_encoder


def main():
    print("=" * 70)
    print("Task 1.2: Soft-Margin Linear SVM from Scratch")
    print("=" * 70)
    
    # Load data
    X, y, class_names, label_encoder = load_data()
    
    # Train-test split (same as classification.py for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiclass SVM
    C = 1.0  # Regularization parameter
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
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix (Test Set):")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Analyze support vectors
    print(f"\n{'='*70}")
    print("SUPPORT VECTOR ANALYSIS")
    print(f"{'='*70}")
    
    analysis = analyze_support_vectors(svm, X_train_scaled, y_train, class_names)
    
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
    
    print(f"\n{'='*70}")
    print("FARTHEST POINTS FROM HYPERPLANE")
    print(f"{'='*70}")
    
    for class_name, farthest in analysis["farthest_points"].items():
        print(f"\n[{class_name} vs Rest classifier]")
        if "positive_class" in farthest:
            p = farthest["positive_class"]
            print(f"  Farthest on +1 side (class={class_name}):")
            print(f"    Index: {p['index']}, Distance: {p['distance']:.4f}, "
                  f"True label: {p['original_label']}")
        if "negative_class" in farthest:
            n = farthest["negative_class"]
            print(f"  Farthest on -1 side (other classes):")
            print(f"    Index: {n['index']}, Distance: {n['distance']:.4f}, "
                  f"True label: {n['original_label']}")
    
    # Compute pairwise distances between support vectors
    print(f"\n{'='*70}")
    print("SUPPORT VECTOR PAIRWISE DISTANCES")
    print(f"{'='*70}")
    
    sv_distances = compute_sv_pairwise_distances(
        X_train_scaled, analysis["global_support_vector_indices"], y_train, class_names
    )
    
    # Show closest pairs (different classes)
    cross_class = sv_distances[~sv_distances["same_class"]]
    print(f"\nClosest support vector pairs from DIFFERENT classes (top 20):")
    print(cross_class.head(20).to_string(index=False))
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        "svm": svm,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "analysis": analysis,
        "sv_pairwise_distances": sv_distances,
        "confusion_matrix": cm,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    
    results_path = RESULTS_DIR / "svm_scratch_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")
    
    # Save summary CSV
    summary_rows = []
    for class_name, info in analysis["per_class"].items():
        summary_rows.append({
            "Classifier": f"{class_name} vs Rest",
            "N_Support_Vectors": info["n_support_vectors"],
            "W_Norm": info["w_norm"],
            "Bias": info["b"],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "svm_scratch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # Save farthest points info
    farthest_rows = []
    for class_name, farthest in analysis["farthest_points"].items():
        if "positive_class" in farthest:
            p = farthest["positive_class"]
            farthest_rows.append({
                "Classifier": f"{class_name} vs Rest",
                "Side": "Positive (+1)",
                "Index": p["index"],
                "Distance": p["distance"],
                "True_Label": p["original_label"],
            })
        if "negative_class" in farthest:
            n = farthest["negative_class"]
            farthest_rows.append({
                "Classifier": f"{class_name} vs Rest",
                "Side": "Negative (-1)",
                "Index": n["index"],
                "Distance": n["distance"],
                "True_Label": n["original_label"],
            })
    
    farthest_df = pd.DataFrame(farthest_rows)
    farthest_path = RESULTS_DIR / "svm_farthest_points.csv"
    farthest_df.to_csv(farthest_path, index=False)
    print(f"Farthest points saved to: {farthest_path}")
    
    # Save cross-class SV distances
    sv_dist_path = RESULTS_DIR / "svm_sv_pairwise_distances.csv"
    sv_distances.to_csv(sv_dist_path, index=False)
    print(f"SV pairwise distances saved to: {sv_dist_path}")
    
    print(f"\n{'='*70}")
    print("DONE - Ready for visual inspection and distance analysis")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

