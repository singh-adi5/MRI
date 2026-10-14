"""Score entities with the trained models.

Three defects are fixed here.

* **The anomaly signal was double counted.** The ML risk score was
  ``0.70 * supervised + 0.30 * anomaly`` and the anomaly score *also* carried
  0.15 of the fusion, so its effective weight was ``0.40 * 0.30 + 0.15 = 0.27``
  against a published 0.15. The blend now comes from ``mri.scoring.ML_BLEND``,
  which is supervised-only; anomaly enters the score once, in the fusion.

* **Scores were batch-relative.** ``normalize()`` min-maxed the anomaly score
  across whatever population was in front of it, so the most anomalous entity
  was pinned to 1.0 by construction on every run and adding one entity moved
  everybody. The raw anomaly value is now mapped through the reference
  distribution frozen at training time.

* **No single-entity path.** :func:`score_one` scores one feature dict with no
  population at all, which is what the API needs.

Scores are stamped ``in_sample`` from the holdout manifest written by
``mri.models.train``. An in-sample score is not a result.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from mri.config import ROOT as BASE_DIR  # repository root
from mri.scoring import (
    ML_BLEND,
    REFERENCE_PATH,
    ReferenceCalibration,
    band_for,
    load_reference,
)

FEATURE_PATH = os.path.join(BASE_DIR, "artifacts", "entity_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
SPLIT_MANIFEST_PATH = os.path.join(MODEL_DIR, "split_manifest.json")

CLF_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
ISO_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def log(msg: str):
    print(f"[score_entities] {msg}")


def entity_key(entity_type, entity_id) -> str:
    return f"{str(entity_type).strip().lower()}:{str(entity_id).strip()}"


def load_holdout_keys() -> set[str]:
    """Keys the models never saw. Empty set means 'unknown', not 'none'."""
    if not os.path.exists(SPLIT_MANIFEST_PATH):
        return set()
    with open(SPLIT_MANIFEST_PATH, encoding="utf-8") as f:
        return set(json.load(f).get("holdout_keys", []))


# --------------------------------------------------------------------------
# Single-entity path (no population required).
# --------------------------------------------------------------------------

_BUNDLE: dict | None = None


def load_bundle(reference_path=REFERENCE_PATH) -> dict:
    """Load models + frozen reference once. Safe to call per request."""
    global _BUNDLE
    if _BUNDLE is None:
        for p in (CLF_PATH, ISO_PATH, COLS_PATH):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required model artifact: {p}")
        _BUNDLE = {
            "clf": joblib.load(CLF_PATH),
            "iso": joblib.load(ISO_PATH),
            "cols": list(joblib.load(COLS_PATH)),
            "reference": load_reference(reference_path),
        }
    return _BUNDLE


def score_one(features: dict, bundle: dict | None = None) -> dict:
    """Produce the ML-side signals for ONE entity from its feature dict.

    Returns ``supervised`` (probability), ``anomaly_raw``, ``anomaly`` (0-1
    against the frozen reference) and ``ml`` (the ML fusion signal). Feed
    ``ml`` and ``anomaly`` into :func:`mri.scoring.score_entity` together with
    the graph and temporal signals to get the fused score for that entity.

    Features not supplied default to 0.0, matching the training-time fill.
    """
    b = bundle or load_bundle()
    row = {c: float(pd.to_numeric(features.get(c, 0.0), errors="coerce") or 0.0) for c in b["cols"]}
    X = pd.DataFrame([row], columns=b["cols"])

    supervised = float(b["clf"].predict_proba(X)[:, 1][0])
    anomaly_raw = float(-b["iso"].score_samples(X)[0])
    anomaly = float(b["reference"].transform_one("anomaly", anomaly_raw))
    ml = ML_BLEND["supervised"] * supervised + ML_BLEND["anomaly"] * anomaly
    return {
        "supervised": supervised,
        "anomaly_raw": anomaly_raw,
        "anomaly": anomaly,
        "ml": float(np.clip(ml, 0.0, 1.0)),
    }


# --------------------------------------------------------------------------
# Batch path.
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")
    for p in (CLF_PATH, ISO_PATH, COLS_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required model artifact: {p}")
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Missing frozen score reference: {REFERENCE_PATH}. Run mri.models.train first."
        )

    df = pd.read_csv(FEATURE_PATH, dtype={"entity_id": str})
    clf = joblib.load(CLF_PATH)
    iso = joblib.load(ISO_PATH)
    feature_cols = list(joblib.load(COLS_PATH))
    reference = ReferenceCalibration.load(REFERENCE_PATH)

    required_cols = {"entity_id", "entity_type", "label"}
    missing_base = required_cols - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing base columns in feature file: {missing_base}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model feature columns in feature file: {missing_features}")

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    log("Scoring supervised model probabilities...")
    supervised_prob = clf.predict_proba(X)[:, 1]

    log("Scoring anomaly model against the frozen training reference...")
    anomaly_raw = -iso.score_samples(X)  # higher = more anomalous
    anomaly_score = reference.transform("anomaly", anomaly_raw)

    staleness = reference.staleness("anomaly", len(df))
    if staleness is not None and (staleness > 2.0 or staleness < 0.5):
        log(
            f"WARNING: population is {staleness:.1f}x the size the anomaly reference was "
            f"frozen on. Retrain (mri.models.train) to rebase the reference."
        )

    # ML risk = supervised only. ML_BLEND['anomaly'] is 0.00 precisely so the
    # anomaly signal is not counted here as well as in the fusion.
    final_score = ML_BLEND["supervised"] * supervised_prob + ML_BLEND["anomaly"] * anomaly_score

    holdout = load_holdout_keys()
    keys = df.apply(lambda r: entity_key(r["entity_type"], r["entity_id"]), axis=1)

    result = df[["entity_id", "entity_type", "label"]].copy()
    result["supervised_score"] = supervised_prob
    result["anomaly_score_raw"] = anomaly_raw
    result["anomaly_score"] = anomaly_score          # calibrated 0-1, frozen reference
    result["final_risk_score"] = np.round(final_score, 6)
    result["risk_band"] = result["final_risk_score"].apply(band_for)
    result["in_sample"] = ~keys.isin(holdout) if holdout else True
    result["scored_against_reference"] = os.path.basename(str(REFERENCE_PATH))

    result = result.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)

    n_oos = int((~result["in_sample"]).sum())
    log(f"Scored entities saved to: {OUTPUT_PATH}")
    log(f"Output shape: {result.shape}")
    log(f"Out-of-sample rows: {n_oos} / {len(result)} (the rest are in-sample)")
    if not holdout:
        log("WARNING: no split manifest found; every row marked in_sample=True.")
    print(result.head(10))


if __name__ == "__main__":
    main()
