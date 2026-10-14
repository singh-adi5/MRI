import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "entity_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
MODEL_REPORT_PATH = os.path.join(ARTIFACT_DIR, "model_report.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")


def log(msg: str):
    print(f"[train_model] {msg}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)

    required_cols = {"entity_id", "entity_type", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")

    feature_cols = [col for col in df.columns if col not in ["entity_id", "entity_type", "label"]]
    if not feature_cols:
        raise ValueError("No model feature columns found.")

    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Defensive numeric handling
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    label_counts = y.value_counts(dropna=False).to_dict()
    log(f"Dataset shape: {df.shape}")
    log(f"Label distribution: {label_counts}")

    if y.nunique() < 2:
        raise ValueError("Training labels contain only one class. Cannot train supervised model.")

    # Stratified split for weak labels
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    log("Training supervised RandomForest model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, zero_division=0)
    print("=== Supervised Model Report ===")
    print(report_text)

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    # Unsupervised anomaly model
    log("Training IsolationForest anomaly model...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X)

    # Save artifacts
    joblib.dump(clf, RISK_MODEL_PATH)
    joblib.dump(iso, ANOMALY_MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    report = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "feature_count": int(len(feature_cols)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": auc,
        "warning": (
            "Metrics are based on weak supervision labels derived from V1 rules. "
            "Use for prioritisation and architecture demonstration, not as final production accuracy."
        )
    }

    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log(f"Saved supervised model: {RISK_MODEL_PATH}")
    log(f"Saved anomaly model: {ANOMALY_MODEL_PATH}")
    log(f"Saved feature columns: {FEATURE_COLS_PATH}")
    log(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    log(f"Saved model report: {MODEL_REPORT_PATH}")


if __name__ == "__main__":
    main()