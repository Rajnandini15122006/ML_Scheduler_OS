import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "data/workloads_realistic.csv"

FEATURES = [
    "cpu_burst",
    "io_burst",
    "priority",
    "previous_runtime",
    "context_switches"
]

TARGET = "workload_type"


def load_data():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    return df, X, y


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def main():

    print("=" * 60)
    print("ML WORKLOAD CLASSIFIER")
    print("=" * 60)

    df, X, y = load_data()

    print("\nDataset shape:")
    print(df.shape)

    print("\nFeatures:")
    print(FEATURES)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = train_model(X_train, y_train)

    print("\nRandom Forest training completed.")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(confusion_matrix(y_test, predictions))

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()