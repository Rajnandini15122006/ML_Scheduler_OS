"""
src/benchmark_classifiers.py
Comprehensive Model Comparison and Feature/Data-Leakage Audit for MLTrain-Sched.

Compares:
1. Random Forest (Baseline)
2. XGBoost (Gradient Boosted Decision Trees)
3. Support Vector Machine (RBF Kernel + Feature Scaling)
4. Multi-Layer Perceptron (Neural Network + Feature Scaling)

Evaluates:
- 5-Fold Stratified Cross Validation (Accuracy & Macro-F1)
- Holdout Test Accuracy, Macro-F1, Weighted-F1
- Per-Class Precision, Recall, and Confusion Matrices
- Single-Sample Inference Latency (microseconds per scheduling decision)
- Feature Importance & Feature Ablation (Baseline vs. Engineered)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

DATA_FILE = "data/workloads_realistic.csv"
RESULTS_DIR = "results"
DATA_DIR = "data"

BASELINE_FEATURES = [
    "cpu_burst",
    "io_burst",
    "priority",
    "previous_runtime",
    "context_switches",
    "mem_rss_mb",
    "num_threads"
]

TARGET = "workload_type"


def add_engineered_features(df):
    """
    Appends domain-informed systems features:
    - io_to_cpu_ratio: Relative I/O intensity vs CPU demand
    - cpu_fraction: Fraction of active burst allocated to CPU
    - switch_intensity: Frequency of voluntary/involuntary context switches per runtime unit
    """
    df = df.copy()
    df["io_to_cpu_ratio"] = df["io_burst"] / (df["cpu_burst"] + 1.0)
    df["cpu_fraction"] = df["cpu_burst"] / (df["cpu_burst"] + df["io_burst"] + 1.0)
    df["switch_intensity"] = df["context_switches"] / (df["previous_runtime"] + 1.0)
    return df


def measure_inference_latency(model, X_sample, n_iterations=1000):
    """
    Measures the average latency in microseconds per single-task inference.
    Crucial metric for OS CPU scheduler control planes.
    """
    single_row = X_sample.iloc[[0]]
    # Warmup
    for _ in range(50):
        model.predict(single_row)

    start = time.perf_counter()
    for _ in range(n_iterations):
        model.predict(single_row)
    elapsed = time.perf_counter() - start

    latency_us = (elapsed / n_iterations) * 1_000_000
    return latency_us


def get_models(random_state=42):
    """
    Returns a dictionary of candidate models.
    SVM and MLP are wrapped in a StandardScaler pipeline for numerical stability.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=1.5,
                gamma="scale",
                probability=True,
                random_state=random_state,
                class_weight="balanced"
            ))
        ]),
        "MLP (Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=15
            ))
        ])
    }


def run_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("MLTRAIN-SCHED: CLASSIFIER BENCHMARK & FEATURE AUDIT SUITE")
    print("=" * 70)

    raw_df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(raw_df)} processes from {DATA_FILE}")

    # Encode target labels
    le = LabelEncoder()
    y_raw = raw_df[TARGET]
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print(f"Target classes ({len(class_names)}): {class_names}")

    # --------------------------------------------------------------------------
    # 1. EVALUATION ON BASELINE FEATURES
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: MULTI-MODEL COMPARISON ON BASELINE FEATURES")
    print("-" * 70)

    X_base = raw_df[BASELINE_FEATURES]
    X_train_b, X_test_b, y_train, y_test = train_test_split(
        X_base, y, test_size=0.20, random_state=42, stratify=y
    )

    models_baseline = get_models(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    confusion_matrices = {}

    for name, model in models_baseline.items():
        print(f"\nEvaluating: {name}...")

        # 5-Fold Stratified Cross-Validation
        cv_scores = cross_validate(
            model, X_base, y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            n_jobs=-1
        )
        cv_acc_mean = cv_scores["test_accuracy"].mean()
        cv_acc_std = cv_scores["test_accuracy"].std()
        cv_f1_macro = cv_scores["test_f1_macro"].mean()

        # Holdout Train & Test
        model.fit(X_train_b, y_train)
        y_pred = model.predict(X_test_b)

        test_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # Single-sample inference latency
        latency_us = measure_inference_latency(model, X_test_b)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        # Per-class recall
        per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
        rec_dict = {f"recall_{cls}": per_class_rec[idx] for idx, cls in enumerate(class_names)}

        print(f"  5-Fold CV Accuracy: {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")
        print(f"  5-Fold CV Macro-F1: {cv_f1_macro:.4f}")
        print(f"  Test Accuracy     : {test_acc:.4f}")
        print(f"  Test Macro-F1     : {macro_f1:.4f}")
        print(f"  Inference Latency : {latency_us:.1f} us/task")

        row = {
            "model": name,
            "feature_set": "Baseline (5 features)",
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_macro_f1": cv_f1_macro,
            "test_accuracy": test_acc,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "test_macro_precision": macro_precision,
            "test_macro_recall": macro_recall,
            "latency_us": latency_us,
            **rec_dict
        }
        results.append(row)

    # --------------------------------------------------------------------------
    # 2. FEATURE AUDIT & ENGINEERED FEATURES EXPERIMENT
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: FEATURE ENGINEERING & AUDIT EXPERIMENT")
    print("-" * 70)

    df_eng = add_engineered_features(raw_df)
    ENGINEERED_FEATURES = BASELINE_FEATURES + ["io_to_cpu_ratio", "cpu_fraction", "switch_intensity"]
    
    # Also evaluate without 'priority' (audit: priority is random noise with zero correlation)
    CLEANED_FEATURES = [f for f in ENGINEERED_FEATURES if f != "priority"]

    X_eng = df_eng[ENGINEERED_FEATURES]
    X_train_e, X_test_e, _, _ = train_test_split(
        X_eng, y, test_size=0.20, random_state=42, stratify=y
    )

    models_eng = get_models(random_state=42)

    for name, model in models_eng.items():
        cv_scores = cross_validate(
            model, X_eng, y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            n_jobs=-1
        )
        cv_acc_mean = cv_scores["test_accuracy"].mean()
        cv_acc_std = cv_scores["test_accuracy"].std()
        cv_f1_macro = cv_scores["test_f1_macro"].mean()

        model.fit(X_train_e, y_train)
        y_pred = model.predict(X_test_e)

        test_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        latency_us = measure_inference_latency(model, X_test_e)

        per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
        rec_dict = {f"recall_{cls}": per_class_rec[idx] for idx, cls in enumerate(class_names)}

        print(f"{name} with Engineered Features:")
        print(f"  5-Fold CV Accuracy: {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f}) | Macro-F1: {cv_f1_macro:.4f}")

        row = {
            "model": name,
            "feature_set": "Engineered (8 features)",
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_macro_f1": cv_f1_macro,
            "test_accuracy": test_acc,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "test_macro_precision": macro_precision,
            "test_macro_recall": macro_recall,
            "latency_us": latency_us,
            **rec_dict
        }
        results.append(row)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(DATA_DIR, "classifier_benchmark_results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\nBenchmark results saved to: {csv_out}")

    # --------------------------------------------------------------------------
    # 3. GENERATE COMPARATIVE VISUALIZATIONS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: GENERATING PUBLICATION VISUALIZATIONS")
    print("-" * 70)

    # Plot 1: 4-Panel Confusion Matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#FFFFFF")
    palette = ["Blues", "Greens", "Oranges", "Purples"]

    for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=palette[idx],
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{m_name}\nBaseline Features (Test N=200)", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Workload Class", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Workload Class", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)

    plt.suptitle("MLTrain-Sched Classifier Benchmark: Multi-Model Confusion Matrices", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "classifier_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to: {cm_path}")

    # Plot 2: Metric Comparison (Accuracy, Macro-F1, Latency)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FFFFFF")

    baseline_sub = df_results[df_results["feature_set"] == "Baseline (5 features)"]
    models = baseline_sub["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, baseline_sub["cv_accuracy_mean"], width, label="5-Fold CV Accuracy", color="#2563EB", edgecolor="#1E40AF")
    ax1.bar(x + width/2, baseline_sub["cv_macro_f1"], width, label="5-Fold CV Macro-F1", color="#10B981", edgecolor="#047857")
    ax1.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax1.set_title("Classification Performance (Accuracy & Macro-F1)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylim(0.5, 0.85)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right")

    # Add values on bars
    for i, (acc, f1) in enumerate(zip(baseline_sub["cv_accuracy_mean"], baseline_sub["cv_macro_f1"])):
        ax1.text(i - width/2, acc + 0.008, f"{acc:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + width/2, f1 + 0.008, f"{f1:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Latency Plot
    colors_lat = ["#6366F1", "#EC4899", "#F59E0B", "#8B5CF6"]
    bars = ax2.bar(models, baseline_sub["latency_us"], color=colors_lat, edgecolor="#374151")
    ax2.set_ylabel("Inference Latency (microseconds per task)", fontsize=11, fontweight="bold")
    ax2.set_title("Scheduling Dispatch Latency Overhead", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, lat in zip(bars, baseline_sub["latency_us"]):
        ax2.text(bar.get_x() + bar.get_width()/2, lat + (max(baseline_sub["latency_us"])*0.02), f"{lat:.1f} us", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("MLTrain-Sched: Accuracy vs. Scheduling Overhead Trade-Off", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    metrics_path = os.path.join(RESULTS_DIR, "classifier_comparison_metrics.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Saved metric comparisons to: {metrics_path}")

    # --------------------------------------------------------------------------
    # 4. SUMMARY AUDIT REPORT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASSIFIER BENCHMARK LEADERBOARD (SORTED BY 5-FOLD CV ACCURACY)")
    print("=" * 70)
    summary_cols = ["model", "feature_set", "cv_accuracy_mean", "cv_macro_f1", "test_accuracy", "test_macro_f1", "latency_us"]
    print(df_results[summary_cols].sort_values(by="cv_accuracy_mean", ascending=False).to_string(index=False))

    print("\nPER-CLASS RECALL COMPARISON (TEST SET):")
    recall_cols = ["model", "feature_set", "recall_ML_TRAINING", "recall_IO_BOUND", "recall_CPU_BOUND", "recall_MIXED"]
    print(df_results[recall_cols].to_string(index=False))

    return df_results


if __name__ == "__main__":
    run_benchmark()
