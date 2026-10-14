"""Train the supervised and anomaly models, and freeze everything downstream needs.

Three things are produced here that did not exist before, and they matter more
than the models themselves:

1. A persisted **holdout manifest** (``models/split_manifest.json``). The split
   used to be thrown away, so every number the product surfaced afterwards was
   an in-sample number dressed up as a result. Scoring now stamps each entity
   ``in_sample`` true/false and evaluation reports both.

2. The IsolationForest is fit on the **training rows only**. It used to be fit
   on the full matrix (``iso.fit(X)``), so the holdout leaked into the
   unsupervised signal and the "out-of-sample" anomaly score was not.

3. A **frozen reference distribution** for the anomaly signal
   (``models/score_reference.json``), fit on the training rows. Scoring maps
   raw anomaly values through this instead of min-maxing whatever batch is in
   front of it, so a score means the same thing from run to run and a single
   entity can be scored with no population at all.

Labels here are weak labels from the V1 rule engine. Ground truth
(``data/raw/ground_truth.csv``) is NEVER read by this module: it exists only to
measure, in ``mri.evaluation``.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mri.config import ROOT as BASE_DIR  # repository root
from mri.scoring import ReferenceCalibration, REFERENCE_PATH

FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "entity_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SPLIT_MANIFEST_PATH = os.path.join(MODEL_DIR, "split_manifest.json")
MODEL_REPORT_PATH = os.path.join(ARTIFACT_DIR, "model_report.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")

TEST_SIZE = 0.25
RANDOM_STATE = 42


def log(msg: str):
    print(f"[train_model] {msg}")


def entity_key(entity_type, entity_id) -> str:
    """Stable join key used by every stage: '<type>:<id>'."""
    return f"{str(entity_type).strip().lower()}:{str(entity_id).strip()}"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH, dtype={"entity_id": str})

    required_cols = {"entity_id", "entity_type", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")

    feature_cols = [c for c in df.columns if c not in ["entity_id", "entity_type", "label"]]
    if not feature_cols:
        raise ValueError("No model feature columns found.")

    X = df[feature_cols].copy()
    y = df["label"].copy()
    keys = df.apply(lambda r: entity_key(r["entity_type"], r["entity_id"]), axis=1)

    # Defensive numeric handling
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    label_counts = y.value_counts(dropna=False).to_dict()
    log(f"Dataset shape: {df.shape}")
    log(f"Label distribution (weak labels from V1 rules): {label_counts}")

    if y.nunique() < 2:
        raise ValueError("Training labels contain only one class. Cannot train supervised model.")

    # Stratified split on the weak labels. The index is carried through so the
    # exact holdout membership can be persisted, not just its size.
    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    log(f"Training supervised RandomForest on {len(X_train)} rows (holdout {len(X_test)})...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    report_text = classification_report(y_test, y_pred, zero_division=0)
    print("=== Supervised Model Report (holdout, weak labels) ===")
    print(report_text)

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    # Unsupervised anomaly model: TRAIN ROWS ONLY. Fitting on the full matrix
    # leaked the holdout into the anomaly signal.
    log(f"Training IsolationForest on {len(X_train)} training rows only...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # Freeze the anomaly reference distribution on the training rows, so the
    # anomaly signal is calibrated against a fixed population rather than
    # min-maxed across whatever batch happens to be scored.
    anomaly_train_raw = -iso.score_samples(X_train)  # higher = more anomalous
    reference = ReferenceCalibration.load_or_empty(REFERENCE_PATH)
    reference.fit_signal(
        "anomaly",
        anomaly_train_raw,
        source=f"IsolationForest(-score_samples) on {len(X_train)} training rows",
    )
    reference.meta.update(
        {
            "anomaly_fitted_rows": int(len(X_train)),
            "feature_count": int(len(feature_cols)),
            "random_state": RANDOM_STATE,
        }
    )
    reference.save(REFERENCE_PATH)

    joblib.dump(clf, RISK_MODEL_PATH)
    joblib.dump(iso, ANOMALY_MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    split_manifest = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stratified_on": "label (V1 weak labels)",
        "n_train": int(len(idx_train)),
        "n_holdout": int(len(idx_test)),
        "train_keys": sorted(keys.iloc[idx_train].tolist()),
        "holdout_keys": sorted(keys.iloc[idx_test].tolist()),
        "note": (
            "Entities in holdout_keys were seen by neither the supervised model nor "
            "the IsolationForest. Scores for them are out-of-sample; every other "
            "score in the product is in-sample and must be read as such."
        ),
    }
    with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    report = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "feature_count": int(len(feature_cols)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "n_train": int(len(idx_train)),
        "n_holdout": int(len(idx_test)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": auc,
        "metrics_measured_on": "holdout rows only",
        "anomaly_fit_rows": int(len(X_train)),
        "warning": (
            "Metrics are against weak supervision labels derived from V1 rules, not "
            "ground truth. They measure agreement with the rule engine, not fraud "
            "detection quality. See outputs/evaluation.json for measurement against "
            "data/raw/ground_truth.csv."
        ),
    }

    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log(f"Saved supervised model: {RISK_MODEL_PATH}")
    log(f"Saved anomaly model: {ANOMALY_MODEL_PATH}")
    log(f"Saved feature columns: {FEATURE_COLS_PATH}")
    log(f"Saved split manifest: {SPLIT_MANIFEST_PATH}")
    log(f"Saved frozen score reference: {REFERENCE_PATH}")
    log(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")
    log(f"Saved model report: {MODEL_REPORT_PATH}")


if __name__ == "__main__":
    main()
