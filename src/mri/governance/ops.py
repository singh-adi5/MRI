import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np


from mri.config import ROOT as BASE_DIR  # repository root
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
MODEL_REPORT_PATH = os.path.join(ARTIFACT_DIR, "model_report.json")
BASELINE_PROFILE_PATH = os.path.join(ARTIFACT_DIR, "baseline_feature_profile.json")

RISK_V2_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")
TEMPORAL_PATH = os.path.join(OUTPUT_DIR, "temporal_risk_scores.csv")
FUSED_V4_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v4.csv")

RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

OPS_DB_PATH = os.path.join(BASE_DIR, "outputs", "ops_registry.db")
PIPELINE_HEALTH_JSON = os.path.join(OUTPUT_DIR, "v6_pipeline_health.json")
DRIFT_REPORT_CSV = os.path.join(OUTPUT_DIR, "v6_feature_drift.csv")


def log(msg: str):
    print(f"[v6_operationalize] {msg}")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def safe_read_json(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_ops_db():
    conn = sqlite3.connect(OPS_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT,
            status TEXT,
            stage TEXT,
            details TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            registry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            registered_ts TEXT,
            model_name TEXT,
            file_path TEXT,
            file_size_bytes INTEGER,
            sha256 TEXT,
            metadata_json TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            drift_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_ts TEXT,
            feature_name TEXT,
            baseline_mean REAL,
            current_mean REAL,
            mean_shift_pct REAL,
            baseline_std REAL,
            current_std REAL,
            std_shift_pct REAL,
            baseline_missing_pct REAL,
            current_missing_pct REAL,
            drift_flag TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS governance_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_ts TEXT,
            artifact_name TEXT,
            artifact_path TEXT,
            artifact_hash TEXT,
            artifact_metadata TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_pipeline_run(status: str, stage: str, details: str):
    conn = sqlite3.connect(OPS_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO pipeline_runs (run_ts, status, stage, details)
        VALUES (?, ?, ?, ?)
        """,
        (utc_now(), status, stage, details)
    )
    conn.commit()
    conn.close()


def register_model_artifact(model_name: str, path: str, metadata: dict):
    file_size = os.path.getsize(path)
    file_hash = sha256_file(path)

    conn = sqlite3.connect(OPS_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_registry (registered_ts, model_name, file_path, file_size_bytes, sha256, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            model_name,
            path,
            file_size,
            file_hash,
            json.dumps(metadata, ensure_ascii=False)
        )
    )
    conn.commit()
    conn.close()


def register_governance_artifact(name: str, path: str, metadata: dict):
    file_hash = sha256_file(path) if os.path.exists(path) else None

    conn = sqlite3.connect(OPS_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO governance_artifacts (artifact_ts, artifact_name, artifact_path, artifact_hash, artifact_metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            name,
            path,
            file_hash,
            json.dumps(metadata, ensure_ascii=False)
        )
    )
    conn.commit()
    conn.close()


def validate_schema(df: pd.DataFrame, required_cols: set, label: str):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def build_feature_profile(df: pd.DataFrame) -> dict:
    numeric_cols = [c for c in df.columns if c not in ["entity_id", "entity_type", "label"]]
    profile = {}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        profile[col] = {
            "mean": float(series.mean()) if not series.isna().all() else 0.0,
            "std": float(series.std()) if not series.isna().all() else 0.0,
            "missing_pct": float(series.isna().mean() * 100.0),
            "min": float(series.min()) if not series.isna().all() else 0.0,
            "max": float(series.max()) if not series.isna().all() else 0.0
        }

    return profile


def pct_shift(baseline, current):
    if baseline == 0:
        return 0.0 if current == 0 else 100.0
    return ((current - baseline) / abs(baseline)) * 100.0


def compute_drift_report(baseline_profile: dict, current_profile: dict) -> pd.DataFrame:
    rows = []
    all_features = sorted(set(baseline_profile.keys()) | set(current_profile.keys()))

    for feat in all_features:
        b = baseline_profile.get(feat, {"mean": 0, "std": 0, "missing_pct": 0})
        c = current_profile.get(feat, {"mean": 0, "std": 0, "missing_pct": 0})

        mean_shift = pct_shift(b.get("mean", 0), c.get("mean", 0))
        std_shift = pct_shift(b.get("std", 0), c.get("std", 0))

        drift_flag = "No"
        if abs(mean_shift) > 25 or abs(std_shift) > 25 or abs(c.get("missing_pct", 0) - b.get("missing_pct", 0)) > 10:
            drift_flag = "Yes"

        rows.append({
            "feature_name": feat,
            "baseline_mean": b.get("mean", 0),
            "current_mean": c.get("mean", 0),
            "mean_shift_pct": round(mean_shift, 4),
            "baseline_std": b.get("std", 0),
            "current_std": c.get("std", 0),
            "std_shift_pct": round(std_shift, 4),
            "baseline_missing_pct": b.get("missing_pct", 0),
            "current_missing_pct": c.get("missing_pct", 0),
            "drift_flag": drift_flag
        })

    return pd.DataFrame(rows).sort_values(["drift_flag", "mean_shift_pct"], ascending=[False, False])


def persist_drift_report(drift_df: pd.DataFrame):
    conn = sqlite3.connect(OPS_DB_PATH)
    cur = conn.cursor()

    for _, row in drift_df.iterrows():
        cur.execute(
            """
            INSERT INTO drift_reports (
                report_ts, feature_name, baseline_mean, current_mean, mean_shift_pct,
                baseline_std, current_std, std_shift_pct,
                baseline_missing_pct, current_missing_pct, drift_flag
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                row["feature_name"],
                float(row["baseline_mean"]),
                float(row["current_mean"]),
                float(row["mean_shift_pct"]),
                float(row["baseline_std"]),
                float(row["current_std"]),
                float(row["std_shift_pct"]),
                float(row["baseline_missing_pct"]),
                float(row["current_missing_pct"]),
                row["drift_flag"]
            )
        )

    conn.commit()
    conn.close()


def main():
    ensure_dirs()
    init_ops_db()

    try:
        insert_pipeline_run("STARTED", "V6", "Starting governance, registry, and drift control checks")

        log("Loading required artifacts...")
        feature_df = safe_read_csv(FEATURE_PATH, "entity features")
        fused_df = safe_read_csv(FUSED_V4_PATH, "final fused risk")
        explain_df = safe_read_csv(EXPLAIN_PATH, "entity explanations")
        graph_df = safe_read_csv(GRAPH_PATH, "graph risk")
        temporal_df = safe_read_csv(TEMPORAL_PATH, "temporal risk")
        risk_v2_df = safe_read_csv(RISK_V2_PATH, "V2 risk output")
        feature_importance_df = safe_read_csv(FEATURE_IMPORTANCE_PATH, "feature importance")
        model_report = safe_read_json(MODEL_REPORT_PATH, "model report")

        log("Validating schemas...")
        validate_schema(feature_df, {"entity_id", "entity_type", "label"}, "entity_features.csv")
        validate_schema(risk_v2_df, {"entity_id", "entity_type", "final_risk_score", "risk_band"}, "final_risk_entities.csv")
        validate_schema(explain_df, {"entity_id", "entity_type", "analyst_summary"}, "entity_explanations.csv")
        validate_schema(graph_df, {"entity_id", "entity_type", "graph_risk_score"}, "graph_risk_scores.csv")
        validate_schema(temporal_df, {"entity_id", "entity_type", "temporal_risk_score"}, "temporal_risk_scores.csv")
        validate_schema(fused_df, {"entity_id", "entity_type", "fused_risk_score", "fused_risk_band"}, "final_fused_risk_v4.csv")
        validate_schema(feature_importance_df, {"feature", "importance"}, "feature_importance.csv")

        log("Registering model artifacts...")
        for model_name, path, meta in [
            ("risk_classifier", RISK_MODEL_PATH, {"stage": "V2 supervised model"}),
            ("anomaly_model", ANOMALY_MODEL_PATH, {"stage": "V2 unsupervised anomaly model"}),
            ("feature_columns", FEATURE_COLS_PATH, {"stage": "feature schema"})
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing model artifact: {path}")
            register_model_artifact(model_name, path, meta)

        log("Registering governance artifacts...")
        for name, path in [
            ("entity_features", FEATURE_PATH),
            ("final_fused_risk_v4", FUSED_V4_PATH),
            ("entity_explanations", EXPLAIN_PATH),
            ("graph_risk_scores", GRAPH_PATH),
            ("temporal_risk_scores", TEMPORAL_PATH),
            ("feature_importance", FEATURE_IMPORTANCE_PATH)
        ]:
            register_governance_artifact(name, path, {"source": "pipeline"})

        log("Building current feature profile...")
        current_profile = build_feature_profile(feature_df)

        if not os.path.exists(BASELINE_PROFILE_PATH):
            log("No baseline profile found. Creating baseline profile now...")
            with open(BASELINE_PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(current_profile, f, indent=2)
            baseline_profile = current_profile
            baseline_created = True
        else:
            with open(BASELINE_PROFILE_PATH, "r", encoding="utf-8") as f:
                baseline_profile = json.load(f)
            baseline_created = False

        log("Computing feature drift...")
        drift_df = compute_drift_report(baseline_profile, current_profile)
        drift_df.to_csv(DRIFT_REPORT_CSV, index=False)
        persist_drift_report(drift_df)

        drift_features = drift_df[drift_df["drift_flag"] == "Yes"]["feature_name"].tolist()

        health_report = {
            "generated_at_utc": utc_now(),
            "pipeline_status": "HEALTHY" if len(drift_features) == 0 else "WARNING",
            "baseline_created_now": baseline_created,
            "entity_counts": {
                "features": int(len(feature_df)),
                "v2_risk": int(len(risk_v2_df)),
                "graph": int(len(graph_df)),
                "temporal": int(len(temporal_df)),
                "fused_v4": int(len(fused_df))
            },
            "risk_band_distribution": fused_df["fused_risk_band"].value_counts(dropna=False).to_dict()
            if "fused_risk_band" in fused_df.columns else {},
            "model_report_summary": model_report,
            "drift_summary": {
                "features_checked": int(len(drift_df)),
                "features_flagged_for_drift": int(len(drift_features)),
                "drifted_features": drift_features[:25]
            },
            "registered_models": [
                os.path.basename(RISK_MODEL_PATH),
                os.path.basename(ANOMALY_MODEL_PATH),
                os.path.basename(FEATURE_COLS_PATH)
            ]
        }

        with open(PIPELINE_HEALTH_JSON, "w", encoding="utf-8") as f:
            json.dump(health_report, f, indent=2)

        register_governance_artifact(
            "v6_pipeline_health",
            PIPELINE_HEALTH_JSON,
            {"status": health_report["pipeline_status"]}
        )
        register_governance_artifact(
            "v6_feature_drift",
            DRIFT_REPORT_CSV,
            {"drift_features": len(drift_features)}
        )

        insert_pipeline_run(
            "SUCCESS",
            "V6",
            f"V6 completed. Drift flagged on {len(drift_features)} features."
        )

        log(f"Saved pipeline health report: {PIPELINE_HEALTH_JSON}")
        log(f"Saved drift report: {DRIFT_REPORT_CSV}")
        log(f"Operational registry DB: {OPS_DB_PATH}")

        print("\n=== V6 HEALTH SUMMARY ===")
        print(json.dumps(health_report, indent=2))

    except Exception as e:
        insert_pipeline_run("FAILED", "V6", str(e))
        raise


if __name__ == "__main__":
    main()