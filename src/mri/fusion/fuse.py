"""Multi-layer risk fusion.

Every constant used here lives in :mod:`mri.scoring`. Nothing numeric is
declared in this file.

Three defects are fixed.

* **Batch-relative scores.** Each signal used to be min-maxed across the
  current population (``normalize()``), which meant the top entity was pinned
  to 1.0 by construction on every run, adding one entity moved everyone else,
  and the fixed band cuts at 0.30/0.55/0.75 compared numbers that were not
  comparable. Signals are now mapped through a reference distribution frozen
  once and persisted at ``models/score_reference.json``. The same entity gets
  the same score on every run and ``mri.scoring.score_entity`` reproduces it
  for a single entity with no population present.

* **Double-counted anomaly.** The anomaly signal is weighted once, at
  ``WEIGHTS['anomaly']``, and ``mri.models.score`` no longer folds it into the
  ML signal as well. The published weights are now the effective weights.

* **Missing entities failed open.** Entities absent from the graph or temporal
  stage were filled with 0 and therefore read as low risk. A missing signal is
  now NaN, the weights are renormalised over the signals that were actually
  observed, and per-entity and per-run coverage is recorded.
"""

import json
import os

import numpy as np
import pandas as pd

from mri.config import ROOT as BASE_DIR  # repository root
from mri.scoring import (
    REFERENCE_PATH,
    SIGNAL_ORDER,
    WEIGHTS,
    ReferenceCalibration,
    band_for,
    fuse_with_coverage,
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ML_RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
GRAPH_RISK_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")
TEMPORAL_RISK_PATH = os.path.join(OUTPUT_DIR, "temporal_risk_scores.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
FUSED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v4.csv")
COVERAGE_PATH = os.path.join(OUTPUT_DIR, "fusion_coverage.json")

#: Set MRI_REFIT_REFERENCE=1 to deliberately rebase the frozen graph/temporal
#: reference on the current population (e.g. after the dataset is regenerated).
REFIT_ENV = "MRI_REFIT_REFERENCE"


def log(msg: str):
    print(f"[risk_fusion_v4] {msg}")


def safe_read_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path, dtype={"entity_id": str})
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


GRAPH_CARRY = [
    "graph_degree", "degree_centrality", "component_size",
    "shared_device_links", "shared_region_links", "transaction_links",
    "graph_risk_score", "graph_risk_band",
]
TEMPORAL_CARRY = [
    "temporal_risk_score", "temporal_risk_band",
    "order_velocity_per_hour", "order_burst_ratio",
    "refund_velocity_per_hour", "login_velocity_per_hour",
    "failed_attempt_density", "seller_order_velocity_per_hour",
    "seller_order_burst_ratio",
]
EXPLAIN_CARRY = ["top_reason_codes", "top_reason_summary", "analyst_summary", "attribution_method"]


def main():
    log("Loading component outputs...")
    ml_df = safe_read_csv(ML_RISK_PATH, "ML risk output")
    graph_df = safe_read_csv(GRAPH_RISK_PATH, "graph risk output")
    temporal_df = safe_read_csv(TEMPORAL_RISK_PATH, "temporal risk output")
    explain_df = safe_read_csv(EXPLAIN_PATH, "explanation output")

    for df in (ml_df, graph_df, temporal_df, explain_df):
        df["entity_id"] = df["entity_id"].astype(str).str.strip()
        df["entity_type"] = df["entity_type"].astype(str).str.strip().str.lower()

    required_ml = {"entity_id", "entity_type", "supervised_score", "anomaly_score", "final_risk_score"}
    required_graph = {"entity_id", "entity_type", "graph_risk_score"}
    required_temporal = {"entity_id", "entity_type", "temporal_risk_score"}
    required_explain = {"entity_id", "entity_type", "top_reason_codes", "top_reason_summary", "analyst_summary"}
    for df, req, label in [
        (ml_df, required_ml, "ML output"),
        (graph_df, required_graph, "Graph output"),
        (temporal_df, required_temporal, "Temporal output"),
        (explain_df, required_explain, "Explanation output"),
    ]:
        if not req.issubset(df.columns):
            raise ValueError(f"{label} missing columns: {req - set(df.columns)}")

    log("Merging ML, graph, temporal and explanation layers...")
    graph_cols = ["entity_id", "entity_type"] + [c for c in GRAPH_CARRY if c in graph_df.columns]
    temporal_cols = ["entity_id", "entity_type"] + [c for c in TEMPORAL_CARRY if c in temporal_df.columns]
    explain_cols = ["entity_id", "entity_type"] + [c for c in EXPLAIN_CARRY if c in explain_df.columns]

    # Outer joins: an entity seen only by the graph stage is still an entity.
    fused = (
        ml_df.merge(graph_df[graph_cols], on=["entity_id", "entity_type"], how="outer")
        .merge(temporal_df[temporal_cols], on=["entity_id", "entity_type"], how="outer")
        .merge(explain_df[explain_cols], on=["entity_id", "entity_type"], how="left")
    )

    # Which stages actually saw each entity. NOT filled with zero: a detector
    # that never saw an entity has not said the entity is safe.
    fused["seen_by_ml"] = fused["final_risk_score"].notna()
    fused["seen_by_graph"] = fused["graph_risk_score"].notna()
    fused["seen_by_temporal"] = fused["temporal_risk_score"].notna()

    numeric_cols = [
        c for c in fused.columns
        if c not in {"entity_id", "entity_type", "graph_risk_band", "temporal_risk_band",
                     "risk_band", "top_reason_codes", "top_reason_summary", "analyst_summary",
                     "attribution_method", "scored_against_reference",
                     "seen_by_ml", "seen_by_graph", "seen_by_temporal", "in_sample"}
    ]
    for col in numeric_cols:
        fused[col] = pd.to_numeric(fused[col], errors="coerce")

    # ---------------------------------------------------------------
    # Frozen reference calibration.
    # ---------------------------------------------------------------
    refit = os.environ.get(REFIT_ENV, "").strip() in {"1", "true", "yes"}
    reference = ReferenceCalibration.load_or_empty(REFERENCE_PATH)
    if "anomaly" not in reference.signals:
        raise FileNotFoundError(
            f"No frozen anomaly reference in {REFERENCE_PATH}. Run mri.models.train first."
        )

    for name, column in (("graph", "graph_risk_score"), ("temporal", "temporal_risk_score")):
        values = fused.loc[fused[column].notna(), column]
        if values.empty:
            raise ValueError(f"No finite {column} values; cannot fuse the {name} signal.")
        existed = reference.has(name)
        reference.ensure_signal(
            name, values, source=f"{column} over {len(values)} entities", refit=refit
        )
        if existed and not refit:
            staleness = reference.staleness(name, len(values))
            if staleness is not None and (staleness > 2.0 or staleness < 0.5):
                log(
                    f"WARNING: {name} population is {staleness:.1f}x the reference it was frozen "
                    f"on ({reference.signals[name].n}). Set {REFIT_ENV}=1 to rebase deliberately."
                )
        else:
            log(f"Froze {name} reference on {len(values)} entities -> {REFERENCE_PATH}")
    reference.save(REFERENCE_PATH)

    # ml is a probability already in [0, 1]; anomaly arrives from mri.models.score
    # already calibrated against the frozen training reference.
    fused["ml_score_norm"] = np.clip(fused["final_risk_score"], 0.0, 1.0)
    fused["anomaly_score_norm"] = np.clip(fused["anomaly_score"], 0.0, 1.0)
    fused["graph_score_norm"] = reference.transform("graph", fused["graph_risk_score"].to_numpy())
    fused["temporal_score_norm"] = reference.transform("temporal", fused["temporal_risk_score"].to_numpy())

    # ---------------------------------------------------------------
    # Fusion with coverage.
    # ---------------------------------------------------------------
    log("Fusing with published weights: " + ", ".join(f"{k}={v:.2f}" for k, v in WEIGHTS.items()))
    results = [
        fuse_with_coverage(
            ml=r.ml_score_norm,
            graph=r.graph_score_norm,
            temporal=r.temporal_score_norm,
            anomaly=r.anomaly_score_norm,
        )
        for r in fused.itertuples()
    ]

    fused["fused_risk_score"] = np.round([r["fused_risk_score"] for r in results], 6)
    fused["fused_risk_band"] = [r["fused_risk_band"] for r in results]
    fused["signal_coverage"] = [r["coverage"] for r in results]
    fused["signals_missing"] = ["|".join(r["signals_missing"]) for r in results]
    for name in SIGNAL_ORDER:
        fused[f"c_{name}"] = [r["contributions"].get(name, np.nan) for r in results]

    for col in ("graph_risk_band", "temporal_risk_band", "risk_band"):
        if col in fused.columns:
            fused[col] = fused[col].fillna("Unknown")

    def _fmt(v):
        return "n/a" if pd.isna(v) else f"{float(v):.3f}"

    fused["fusion_summary"] = fused.apply(
        lambda row: (
            f"{str(row['entity_type']).title()} {row['entity_id']} has a fused risk score of "
            f"{_fmt(row['fused_risk_score'])} ({row['fused_risk_band']}) on "
            f"{row['signal_coverage']:.0%} signal coverage. "
            f"ML={_fmt(row['final_risk_score'])}, Graph={_fmt(row['graph_risk_score'])}, "
            f"Temporal={_fmt(row['temporal_risk_score'])}, Anomaly={_fmt(row['anomaly_score'])}"
            + (f". Missing: {row['signals_missing']}" if row["signals_missing"] else "")
            + f". Primary cues: {row['top_reason_summary'] if pd.notna(row.get('top_reason_summary')) else 'n/a'}."
        ),
        axis=1,
    )

    fused = fused.sort_values("fused_risk_score", ascending=False, na_position="last").reset_index(drop=True)
    fused.to_csv(FUSED_OUTPUT_PATH, index=False)

    coverage = {
        "entities": int(len(fused)),
        "weights": WEIGHTS,
        "per_signal_coverage": {
            "ml": float(fused["seen_by_ml"].mean()),
            "graph": float(fused["seen_by_graph"].mean()),
            "temporal": float(fused["seen_by_temporal"].mean()),
            "anomaly": float(fused["anomaly_score_norm"].notna().mean()),
        },
        "entities_missing_a_signal": int((fused["signal_coverage"] < 1.0).sum()),
        "entities_with_no_signal": int((fused["signal_coverage"] == 0.0).sum()),
        "mean_coverage": float(fused["signal_coverage"].mean()),
        "reference_frozen_on": {
            name: {"n": ref.n, "method": ref.method, "fitted_at": ref.fitted_at}
            for name, ref in reference.signals.items()
        },
        "band_counts": fused["fused_risk_band"].value_counts().to_dict(),
    }
    with open(COVERAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    log(f"Saved fused output to: {FUSED_OUTPUT_PATH}")
    log(f"Saved coverage report to: {COVERAGE_PATH}")
    log(
        f"Coverage: graph {coverage['per_signal_coverage']['graph']:.1%}, "
        f"temporal {coverage['per_signal_coverage']['temporal']:.1%}; "
        f"{coverage['entities_missing_a_signal']} entities missing at least one signal"
    )
    print(
        fused[
            ["entity_id", "entity_type", "final_risk_score", "graph_risk_score",
             "temporal_risk_score", "anomaly_score", "fused_risk_score",
             "fused_risk_band", "signal_coverage"]
        ].head(10)
    )


if __name__ == "__main__":
    main()
