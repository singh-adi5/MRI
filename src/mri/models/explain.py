"""Per-entity explanations that are actually about that entity, and actually true.

What was wrong
--------------
``classify_reason()`` chose its wording from the feature *name* alone and never
looked at which way the value deviated. An account with zero logins and zero
devices was told it had "High device concentration detected
(unique_devices=0.000)" and "Geographic activity variance observed
(unique_regions=0.000)" - the absence of a signal reported as the presence of
it. Reasons were then ranked by ``abs(z) * importance`` where ``importance`` is
a single global constant per feature, so only ``|z|`` varied: the ranking was
a peer-group outlier list, not a local explanation, and the sign was discarded
exactly where it carried the meaning.

What it does now
----------------
* **Local attribution.** Each entity's supervised probability is decomposed
  exactly into per-feature contributions by walking its own decision path
  through every tree in the forest and accumulating the change in the
  predicted class proportion at each split
  (``prediction = bias + sum(contributions)``, exact, verified at runtime).
  SHAP is not used: ``shap`` is not a dependency of this project and
  ``sklearn`` has no TreeExplainer, so pulling it in would add a heavy
  dependency. The decision-path decomposition is computed from
  ``sklearn.tree`` internals only and is genuinely per-entity - two entities
  with the same feature value get different contributions if they took
  different paths.

* **Signed deviation, direction-gated.** ``z`` keeps its sign. A reason is
  emitted only when the deviation points the way that actually indicates risk
  for that feature (``RISK_DIRECTION``), and the wording says "unusually low"
  when that is what is meant. A feature whose deviation points the safe way is
  dropped, never reworded into a threat.
"""

import os

import numpy as np
import pandas as pd

from mri.config import ROOT as BASE_DIR  # repository root

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")
IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
CLF_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

#: A deviation only counts as a reason if it points this way.
#:   +1  high values indicate risk
#:   -1  low values indicate risk
#:    0  both tails are meaningful, worded per side
RISK_DIRECTION: dict[str, int] = {
    # Refund abuse
    "refund_rate": +1,
    "total_refunds": +1,
    # Account takeover / credential stuffing
    "avg_failed_attempts": +1,
    "max_failed_attempts": +1,
    "unique_devices": +1,
    "unique_regions": +1,
    "login_events": +1,
    # Bust-out / velocity
    "order_spike_ratio": +1,
    "max_daily_orders": +1,
    "avg_daily_orders": +1,
    "total_orders": +1,
    "total_revenue": +1,
    "total_spend": +1,
    "avg_order_value": 0,
    # Rating farming
    "rating_count": +1,
    "total_ratings": +1,
    "rating_std": -1,       # a farmed rating profile is suspiciously uniform
    "avg_rating": 0,
    "avg_rating_given": 0,
    # Collusive rings: a narrow counterparty set at high volume
    "unique_customers": -1,
    "unique_sellers": -1,
    # New accounts carry more risk
    "account_age_days": -1,
}

#: Wording per feature and per side. ``(high_side, low_side)``; ``None`` means
#: that side is not a risk statement and no reason is emitted for it.
REASON_TEXT: dict[str, tuple[str | None, str | None]] = {
    "refund_rate": ("Refund rate far above its peer group", None),
    "total_refunds": ("Unusually high number of refunds", None),
    "avg_failed_attempts": ("Sustained failed-login attempts above peers", None),
    "max_failed_attempts": ("Peak failed-login attempts above peers", None),
    "unique_devices": ("Logs in from an unusually large number of distinct devices", None),
    "unique_regions": ("Logins span an unusually large number of distinct regions", None),
    "login_events": ("Login volume well above peers", None),
    "order_spike_ratio": ("Daily order volume spikes far above this entity's own baseline", None),
    "max_daily_orders": ("Peak single-day order count above peers", None),
    "avg_daily_orders": ("Sustained daily order rate above peers", None),
    "total_orders": ("Order volume well above peers", None),
    "total_revenue": ("Revenue well above peers", None),
    "total_spend": ("Spend well above peers", None),
    "avg_order_value": ("Order values unusually high for its peer group",
                        "Order values unusually low for its peer group"),
    "rating_count": ("Receives an unusually high number of ratings", None),
    "total_ratings": ("Submits an unusually high number of ratings", None),
    "rating_std": (None, "Ratings are unusually uniform (near-zero spread), consistent with farming"),
    "avg_rating": ("Average rating unusually high for its peer group",
                   "Average rating unusually low for its peer group"),
    "avg_rating_given": ("Gives uniformly high ratings relative to peers",
                         "Gives uniformly low ratings relative to peers"),
    "unique_customers": (None, "Serves an unusually narrow set of customers for its order volume"),
    "unique_sellers": (None, "Transacts with an unusually narrow set of sellers for its order volume"),
    "account_age_days": (None, "Account is unusually new for its peer group"),
}

#: Below this the deviation is noise, not a finding.
Z_MIN = 1.0
MAX_REASONS = 3


def log(msg: str):
    print(f"[explain_entities] {msg}")


# --------------------------------------------------------------------------
# Local attribution.
# --------------------------------------------------------------------------

def forest_contributions(clf, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact per-entity, per-feature contributions to the class-1 probability.

    Decomposes each tree's prediction along the decision path the entity
    actually takes: at every split the change in the node's class-1 proportion
    is attributed to the feature that split there. Averaged over the forest,
    ``bias + contributions.sum(axis=1) == clf.predict_proba(X)[:, 1]`` exactly.

    Returns ``(contributions [n_samples, n_features], bias)``.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    total = np.zeros((n_samples, n_features), dtype=np.float64)
    bias = 0.0

    for est in clf.estimators_:
        tree = est.tree_
        values = tree.value  # (n_nodes, n_outputs, n_classes)
        counts = values[:, 0, :]
        totals = counts.sum(axis=1)
        # tree_.value holds class fractions in current sklearn and weighted
        # counts in older builds; normalising covers both.
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(totals > 0, counts[:, 1] / np.where(totals > 0, totals, 1.0), 0.0)

        left, right = tree.children_left, tree.children_right
        internal = np.flatnonzero(left != -1)
        if internal.size == 0:
            bias += float(p1[0])
            continue

        children = np.concatenate([left[internal], right[internal]])
        parents = np.concatenate([internal, internal])
        feats = tree.feature[parents]
        deltas = p1[children] - p1[parents]

        # M[child_node, feature] = delta on the edge entering that node. Each
        # non-root node has exactly one parent, so one entry per node.
        M = np.zeros((tree.node_count, n_features), dtype=np.float64)
        M[children, feats] = deltas

        # A node is on the path iff it is reached; the edge into it is taken
        # iff the node is on the path. One sparse product per tree.
        indicator = est.decision_path(X)
        total += indicator @ M
        bias += float(p1[0])

    n_trees = len(clf.estimators_)
    return total / n_trees, bias / n_trees


# --------------------------------------------------------------------------
# Deviation.
# --------------------------------------------------------------------------

def peer_zscores(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Signed z of every value against its own entity_type peer group."""
    out = pd.DataFrame(index=feature_df.index, columns=feature_cols, dtype=float)
    for etype, grp in feature_df.groupby("entity_type"):
        vals = grp[feature_cols].apply(pd.to_numeric, errors="coerce")
        mean = vals.mean()
        std = vals.std()
        # A constant feature within a peer group carries no information:
        # NaN here keeps it out of the reason list rather than scoring it 0.
        std = std.where(std > 0)
        out.loc[grp.index, feature_cols] = ((vals - mean) / std).to_numpy()
    return out


def reason_for(feature: str, z: float) -> str | None:
    """Wording for a deviation, or None if this deviation is not a risk signal."""
    direction = RISK_DIRECTION.get(feature, 0)
    high_text, low_text = REASON_TEXT.get(feature, (None, None))
    if z > 0:
        if direction < 0:
            return None  # deviation points the safe way
        return high_text or f"{feature} unusually high for its peer group"
    if z < 0:
        if direction > 0:
            return None  # the signal is ABSENT; absence is not presence
        return low_text or f"{feature} unusually low for its peer group"
    return None


def build_reasons(
    feature: str, value: float, z: float, contribution: float
) -> dict | None:
    if not np.isfinite(z) or abs(z) < Z_MIN:
        return None
    text = reason_for(feature, z)
    if text is None:
        return None
    return {
        "feature": feature,
        "value": float(value),
        "zscore": float(z),
        "contribution": float(contribution),
        "direction": "high" if z > 0 else "low",
        "reason_text": f"{text} ({feature}={value:.3f}, z={z:+.2f}, contribution={contribution:+.3f})",
    }


def main():
    for path, label in [
        (FEATURE_PATH, "features file"),
        (RISK_PATH, "risk file"),
        (IMPORTANCE_PATH, "importance file"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    feature_df = pd.read_csv(FEATURE_PATH, dtype={"entity_id": str})
    risk_df = pd.read_csv(RISK_PATH, dtype={"entity_id": str})
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    if feature_df.empty or risk_df.empty or importance_df.empty:
        raise ValueError("One or more required input files are empty.")

    # Model feature space; fall back to the importance file if the model is absent.
    if os.path.exists(COLS_PATH):
        import joblib

        feature_cols = [c for c in joblib.load(COLS_PATH) if c in feature_df.columns]
    else:
        feature_cols = [c for c in importance_df["feature"] if c in feature_df.columns]

    X = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    z_df = peer_zscores(feature_df, feature_cols)

    # Local contributions, with a direction-correct global fallback.
    attribution = "decision-path contribution (exact, per-entity)"
    contrib = None
    if os.path.exists(CLF_PATH):
        import joblib

        clf = joblib.load(CLF_PATH)
        contrib_arr, bias = forest_contributions(clf, X.to_numpy())
        reconstructed = bias + contrib_arr.sum(axis=1)
        # DataFrame here: the forest was fit with feature names, the individual
        # trees were not. Mixing the two raises a sklearn warning for nothing.
        actual = clf.predict_proba(X)[:, 1]
        max_err = float(np.max(np.abs(reconstructed - actual))) if len(actual) else 0.0
        log(f"Attribution check: max |bias + sum(contributions) - predict_proba| = {max_err:.2e}")
        if max_err > 1e-6:
            log("WARNING: decomposition did not reconcile; falling back to signed z x importance.")
        else:
            contrib = pd.DataFrame(contrib_arr, index=feature_df.index, columns=feature_cols)
    if contrib is None:
        attribution = "signed z x global importance (fallback: no reconciled model attribution)"
        imp = importance_df.set_index("feature")["importance"]
        contrib = z_df[feature_cols].mul(
            [float(imp.get(c, 0.0)) for c in feature_cols], axis=1
        )

    feature_df = feature_df.assign(
        _key=feature_df["entity_type"].str.lower() + ":" + feature_df["entity_id"].astype(str)
    )
    lookup = {k: i for i, k in enumerate(feature_df["_key"])}

    rows = []
    for _, row in risk_df.iterrows():
        entity_id = str(row["entity_id"])
        entity_type = str(row["entity_type"])
        key = f"{entity_type.lower()}:{entity_id}"
        idx = lookup.get(key)

        final_risk_score = float(row.get("final_risk_score", np.nan))
        supervised_score = float(row.get("supervised_score", np.nan))
        anomaly_score = float(row.get("anomaly_score", np.nan))
        risk_band = row.get("risk_band", "Unknown")

        candidates: list[dict] = []
        if idx is not None:
            pos = feature_df.index[idx]
            for feature in feature_cols:
                reason = build_reasons(
                    feature,
                    float(X.at[pos, feature]),
                    float(z_df.at[pos, feature]),
                    float(contrib.at[pos, feature]),
                )
                if reason is not None:
                    candidates.append(reason)
            # Rank by how much this entity's own path pushed the score up.
            candidates.sort(key=lambda r: r["contribution"], reverse=True)
            # Only signals that raise the score are reasons to investigate.
            positive = [c for c in candidates if c["contribution"] > 0]
            candidates = positive or candidates[: MAX_REASONS]

        top = candidates[:MAX_REASONS]
        if top:
            top_codes = " | ".join(r["feature"] for r in top)
            top_text = " | ".join(r["reason_text"] for r in top)
        else:
            top_codes = ""
            top_text = "No feature deviates from its peer group in a direction that indicates risk"

        analyst_summary = (
            f"{entity_type.title()} {entity_id} scored {final_risk_score:.3f} on the ML signal "
            f"({risk_band}); supervised {supervised_score:.3f}, anomaly {anomaly_score:.3f}. "
            f"Attribution: {attribution}. "
            f"Primary investigation signals: {top_text}."
        )

        rows.append(
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "risk_band": risk_band,
                "final_risk_score": final_risk_score,
                "supervised_score": supervised_score,
                "anomaly_score": anomaly_score,
                "reason_count": len(top),
                "attribution_method": attribution,
                "top_reason_codes": top_codes,
                "top_reason_summary": top_text,
                "analyst_summary": analyst_summary,
            }
        )

    explain_df = pd.DataFrame(rows)
    explain_df = explain_df.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    explain_df.to_csv(EXPLAIN_PATH, index=False)

    n_empty = int((explain_df["reason_count"] == 0).sum())
    log(f"Saved explanations to: {EXPLAIN_PATH}")
    log(f"Attribution: {attribution}")
    log(f"Entities with no risk-direction deviation: {n_empty} / {len(explain_df)}")
    print(explain_df[["entity_id", "entity_type", "final_risk_score", "top_reason_summary"]].head(5))


if __name__ == "__main__":
    main()
