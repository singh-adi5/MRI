import os
import joblib
import pandas as pd
import numpy as np

from mri.config import ROOT as BASE_DIR  # repository root
FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "entity_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")


def log(msg: str):
    print(f"[score_entities] {msg}")


def normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def assign_risk_band(score: float) -> str:
    if score >= 0.60:
        return "High"
    elif score >= 0.30:
        return "Medium"
    return "Low"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

    clf_path = os.path.join(MODEL_DIR, "risk_classifier.pkl")
    iso_path = os.path.join(MODEL_DIR, "anomaly_model.pkl")
    cols_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    for p in [clf_path, iso_path, cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required model artifact: {p}")

    df = pd.read_csv(FEATURE_PATH)
    clf = joblib.load(clf_path)
    iso = joblib.load(iso_path)
    feature_cols = joblib.load(cols_path)

    required_cols = {"entity_id", "entity_type", "label"}
    missing_base = required_cols - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing base columns in feature file: {missing_base}")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model feature columns in feature file: {missing_features}")

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    log("Scoring supervised model probabilities...")
    supervised_prob = clf.predict_proba(X)[:, 1]

    log("Scoring anomaly model...")
    anomaly_raw = -iso.score_samples(X)   # higher = more anomalous
    anomaly_score = normalize(pd.Series(anomaly_raw, index=df.index))

    # Fusion weights
    supervised_weight = 0.70
    anomaly_weight = 0.30
    final_score = (supervised_weight * supervised_prob) + (anomaly_weight * anomaly_score)

    result = df[["entity_id", "entity_type", "label"]].copy()
    result["supervised_score"] = supervised_prob
    result["anomaly_score"] = anomaly_score
    result["final_risk_score"] = final_score.round(6)
    result["risk_band"] = result["final_risk_score"].apply(assign_risk_band)

    result = result.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)

    log(f"Scored entities saved to: {OUTPUT_PATH}")
    log(f"Output shape: {result.shape}")
    print(result.head(10))


if __name__ == "__main__":
    main()