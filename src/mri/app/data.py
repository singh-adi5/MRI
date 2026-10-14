"""Data access for the workbench: reads pipeline outputs, builds them on first launch."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from mri.config import ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR, ROOT

FUSED_PATH = OUTPUT_DIR / "final_fused_risk_v4.csv"
FEATURES_PATH = ARTIFACT_DIR / "entity_features.csv"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
REPORT_PATH = ARTIFACT_DIR / "model_report.json"
HEALTH_PATH = OUTPUT_DIR / "v6_pipeline_health.json"
DRIFT_PATH = OUTPUT_DIR / "v6_feature_drift.csv"
REGISTRY_DB = OUTPUT_DIR / "ops_registry.db"

# Fusion definition, mirrored from mri.fusion.fuse (weights) and mri.models.score (ML blend).
ML_SUPERVISED_WEIGHT = 0.70
ML_ANOMALY_WEIGHT = 0.30
BANDS = ["Low", "Medium", "High", "Critical"]  # ascending severity
BAND_THRESHOLDS = {"Medium": 0.30, "High": 0.55, "Critical": 0.75}  # fused-score cut-offs


def band_for(score: float) -> str:
    if score >= BAND_THRESHOLDS["Critical"]:
        return "Critical"
    if score >= BAND_THRESHOLDS["High"]:
        return "High"
    if score >= BAND_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


@st.cache_resource(show_spinner="First launch: building the dataset and scoring entities (about 15 s)...")
def ensure_pipeline_outputs() -> bool:
    """Generated files are not committed; run the pipeline once if they are missing."""
    if not FUSED_PATH.exists():
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        # Fixed argv, no shell, no user input.
        subprocess.run(  # nosec B603
            [sys.executable, "-m", "mri.pipeline", "--no-ui"], cwd=ROOT, env=env, check=True
        )
    return True


@st.cache_data
def load_fused() -> pd.DataFrame:
    df = pd.read_csv(FUSED_PATH, dtype={"entity_id": str})
    df["key"] = df["entity_type"].str.lower() + "_" + df["entity_id"]
    df["label_text"] = df["entity_type"].str.title() + " " + df["entity_id"]
    # Additive contribution of each signal: weight x normalised score. The four sum to the fused score.
    from mri.app.theme import SIGNALS

    for s in SIGNALS:
        df[f"c_{s['key']}"] = s["weight"] * df[s["norm"]]
    df["rank"] = np.arange(1, len(df) + 1)
    df["fused_risk_band"] = pd.Categorical(df["fused_risk_band"], categories=BANDS, ordered=True)
    return df


@st.cache_data
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, dtype={"entity_id": str})
    df["key"] = df["entity_type"].str.lower() + "_" + df["entity_id"]
    return df.set_index("key")


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE_PATH).sort_values("importance", ascending=False).reset_index(drop=True)


@st.cache_data
def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_health() -> dict:
    return json.loads(HEALTH_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_drift() -> pd.DataFrame:
    return pd.read_csv(DRIFT_PATH)


@st.cache_data
def load_registry() -> dict[str, pd.DataFrame]:
    """Read-only view of the governance registry."""
    out: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(f"file:{REGISTRY_DB}?mode=ro", uri=True) as conn:
        for table in ("model_registry", "pipeline_runs", "governance_artifacts"):
            out[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)  # nosec B608 (fixed table names)
    return out


@st.cache_resource
def load_models():
    """Models are produced by this app's own pipeline run, so loading them is trusted."""
    clf = joblib.load(MODEL_DIR / "risk_classifier.pkl")
    iso = joblib.load(MODEL_DIR / "anomaly_model.pkl")
    cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    feats = load_features()
    X = feats[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    raw = -iso.score_samples(X)
    return {"clf": clf, "iso": iso, "cols": list(cols), "anom_min": float(raw.min()), "anom_max": float(raw.max())}


def score_vector(values: dict) -> dict:
    """Score one feature vector with the trained models. Returns the ML-side scores."""
    m = load_models()
    X = pd.DataFrame([{c: float(values.get(c, 0.0)) for c in m["cols"]}])
    sup = float(m["clf"].predict_proba(X)[:, 1][0])
    raw = float(-m["iso"].score_samples(X)[0])
    span = m["anom_max"] - m["anom_min"]
    anom = float(np.clip((raw - m["anom_min"]) / span, 0, 1)) if span else 0.0
    ml = ML_SUPERVISED_WEIGHT * sup + ML_ANOMALY_WEIGHT * anom
    return {"supervised": sup, "anomaly": anom, "ml": ml}


def fused_from_ml(ml: float, anomaly: float, graph_norm: float, temporal_norm: float, fused: pd.DataFrame) -> float:
    """Re-run the v4 fusion for one entity with graph and temporal held at their pipeline values."""
    from mri.app.theme import WEIGHTS

    lo, hi = float(fused["final_risk_score"].min()), float(fused["final_risk_score"].max())
    ml_norm = float(np.clip((ml - lo) / (hi - lo), 0, 1)) if hi > lo else 0.0
    return (
        WEIGHTS["ml"] * ml_norm
        + WEIGHTS["graph"] * graph_norm
        + WEIGHTS["temporal"] * temporal_norm
        + WEIGHTS["anomaly"] * float(np.clip(anomaly, 0, 1))
    )
