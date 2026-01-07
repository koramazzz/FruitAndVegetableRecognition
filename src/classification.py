"""Benchmark script for Assignment 2 Task 1 (question 1).

Trains 7 classifiers on precomputed features (dataset/processed/X_final.npy, y_final.npy),
runs a small hyperparameter search, and reports training times/accuracies.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results"

X_PATH = DATA_DIR / "X_final.npy"
Y_PATH = DATA_DIR / "y_final.npy"

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


def build_model_configs() -> List[Dict]:
    """Model and parameter grid definitions."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Note: Features are already in [0,1]; still add StandardScaler for models
    # that benefit (LR, SVM, kNN).
    configs = [
        {
            "name": "LogReg (linear)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=400, n_jobs=-1)),
                ]
            ),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
            "cv": cv,
        },
        {
            "name": "LogReg + RBF Features",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    # Kernel approximation with RBFSampler to reduce computation cost
                    ("rff", RBFSampler(random_state=RANDOM_STATE)),
                    ("clf", LogisticRegression(max_iter=400, n_jobs=-1)),
                ]
            ),
            "param_grid": {
                "rff__gamma": [0.01, 0.1, 1.0],
                "rff__n_components": [300, 800],
                "clf__C": [1.0, 10.0],
            },
            "cv": cv,
        },
        {
            "name": "SVM (linear)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="linear")),
                ]
            ),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
            },
            "cv": cv,
        },
        {
            "name": "SVM (RBF kernel)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC()),
                ]
            ),
            "param_grid": {
                "clf__C": [0.5, 1.0, 5.0],
                "clf__gamma": ["scale", 0.01, 0.05],
                "clf__kernel": ["rbf"],
            },
            "cv": cv,
        },
        {
            "name": "kNN",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 9],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["minkowski"],
            },
            "cv": cv,
        },
        {
            "name": "Naive Bayes (Gaussian)",
            "estimator": Pipeline(
                [
                    ("clf", GaussianNB()),
                ]
            ),
            "param_grid": {
                "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
            "cv": cv,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 400],
                "max_depth": [None, 20, 40],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2],
            },
            "cv": cv,
        },
    ]
    return configs


def run_benchmark():
    X, y, class_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

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

        start = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        best_est = search.best_estimator_
        y_pred = best_est.predict(X_test)
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
    out_csv = RESULTS_DIR / "classification_benchmark.csv"
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
    plot_path = RESULTS_DIR / "classification_benchmark.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_benchmark()
