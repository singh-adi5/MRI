"""Measure the pipeline against ground truth.

``data/raw/ground_truth.csv`` is read HERE AND NOWHERE ELSE. It is a
measurement instrument, not a feature and not a training label: training
labels come from the V1 rule engine (weak supervision). This module asserts
that separation at runtime - if the weak labels agree with ground truth
suspiciously well, it says so loudly rather than reporting a flattering number.

Produces
--------
``outputs/evaluation.json``
    precision@k and recall@k over a k sweep; lift against random selection and
    against the rules-only ranking; ROC AUC and average precision; per-pattern
    recall; alert rate at every band cut; the confusion matrix at the shipped
    0.75 cut; and a ranker comparison (fused / ML-only / rules-only / random).
    Every block is reported twice: over all entities, and over the holdout
    only, because an in-sample number is not a result.

``outputs/lift_curve.csv``
    Long format: one row per (ranker, k) with precision, recall and lift.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mri.config import ROOT as BASE_DIR
from mri.scoring import BANDS, BAND_THRESHOLDS, WEIGHTS

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

GROUND_TRUTH_PATH = os.path.join(RAW_DIR, "ground_truth.csv")
FUSED_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v4.csv")
RULES_PATH = os.path.join(OUTPUT_DIR, "flagged_entities.csv")
EVALUATION_PATH = os.path.join(OUTPUT_DIR, "evaluation.json")
LIFT_CURVE_PATH = os.path.join(OUTPUT_DIR, "lift_curve.csv")

K_SWEEP = (10, 25, 50, 100, 200, 500)
ALERT_THRESHOLD = BAND_THRESHOLDS["Critical"]  # the shipped 0.75 cut
KNOWN_PATTERNS = ("bust_out", "collusive_ring", "refund_abuse", "ato", "rating_farm")

#: Rankers compared. Each maps to the column holding its score.
RANKERS = {
    "fused": "fused_risk_score",
    "ml_only": "ml_score_norm",
    "rules_only": "rules_score",
}


def log(msg: str):
    print(f"[evaluation] {msg}")


def _key(etype, eid) -> str:
    return f"{str(etype).strip().lower()}:{str(eid).strip()}"


# --------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------

def load_ground_truth() -> pd.DataFrame:
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(
            f"Ground truth not found: {GROUND_TRUTH_PATH}. This file is produced by the "
            f"data stage and is used for measurement only."
        )
    gt = pd.read_csv(GROUND_TRUTH_PATH, dtype={"entity_id": str})
    required = {"entity_type", "entity_id", "is_fraud", "fraud_pattern"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"ground_truth.csv missing columns: {missing}")
    gt["entity_type"] = gt["entity_type"].astype(str).str.strip().str.lower()
    gt["entity_id"] = gt["entity_id"].astype(str).str.strip()
    gt["is_fraud"] = pd.to_numeric(gt["is_fraud"], errors="coerce").fillna(0).astype(int)
    gt["fraud_pattern"] = gt["fraud_pattern"].fillna("").astype(str).str.strip().str.lower()
    gt["key"] = [_key(t, i) for t, i in zip(gt["entity_type"], gt["entity_id"])]
    dupes = int(gt["key"].duplicated().sum())
    if dupes:
        log(f"WARNING: ground truth has {dupes} duplicate entity rows; keeping the first of each.")
        gt = gt.drop_duplicates(subset="key", keep="first")
    return gt


def load_rules_scores() -> pd.Series:
    """Rules-only ranking signal: the V1 rule engine's risk points."""
    if not os.path.exists(RULES_PATH):
        log(f"WARNING: {RULES_PATH} not found; the rules-only baseline will be all zero.")
        return pd.Series(dtype=float)
    rules = pd.read_csv(RULES_PATH, dtype={"entity_id": str})
    if rules.empty or "risk_score" not in rules.columns:
        return pd.Series(dtype=float)
    keys = [_key(t, i) for t, i in zip(rules["entity_type"], rules["entity_id"])]
    return pd.Series(
        pd.to_numeric(rules["risk_score"], errors="coerce").fillna(0).to_numpy(), index=keys
    ).groupby(level=0).max()


def build_frame() -> tuple[pd.DataFrame, dict]:
    """Join scores to ground truth. Returns the frame and a join diagnostic."""
    if not os.path.exists(FUSED_PATH):
        raise FileNotFoundError(f"Missing fused output: {FUSED_PATH}. Run mri.fusion.fuse first.")

    scored = pd.read_csv(FUSED_PATH, dtype={"entity_id": str})
    scored["entity_type"] = scored["entity_type"].astype(str).str.strip().str.lower()
    scored["entity_id"] = scored["entity_id"].astype(str).str.strip()
    scored["key"] = [_key(t, i) for t, i in zip(scored["entity_type"], scored["entity_id"])]

    gt = load_ground_truth()
    df = gt.merge(
        scored.drop(columns=["entity_type", "entity_id"]), on="key", how="left", suffixes=("", "_scored")
    )

    rules = load_rules_scores()
    df["rules_score"] = df["key"].map(rules).fillna(0.0)

    for col in RANKERS.values():
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # An entity in ground truth that the pipeline never scored is a miss, not a
    # gap in the measurement: score it at the floor so it counts against recall.
    unscored = int(df["fused_risk_score"].isna().sum())
    for col in RANKERS.values():
        df[col] = df[col].fillna(0.0)

    if "in_sample" in df.columns:
        df["in_sample"] = df["in_sample"].fillna(True).astype(bool)
    else:
        df["in_sample"] = True

    diagnostics = {
        "ground_truth_entities": int(len(gt)),
        "scored_entities": int(len(scored)),
        "ground_truth_entities_not_scored": unscored,
        "scored_entities_not_in_ground_truth": int(len(set(scored["key"]) - set(gt["key"]))),
        "holdout_entities": int((~df["in_sample"]).sum()),
    }
    return df, diagnostics


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

def _order(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Descending score order with a deterministic tiebreak, so ties never
    silently favour a ranker by accident of row order."""
    tmp = df[[score_col, "key"]].copy()
    tmp = tmp.sort_values([score_col, "key"], ascending=[False, True], kind="mergesort")
    return tmp.index.to_numpy()


def precision_recall_at_k(y: np.ndarray, order: np.ndarray, k: int) -> dict:
    k = min(int(k), len(order))
    if k <= 0:
        return {"k": 0, "true_positives": 0, "precision_at_k": 0.0, "recall_at_k": 0.0}
    hits = int(y[order[:k]].sum())
    total = int(y.sum())
    return {
        "k": k,
        "true_positives": hits,
        "precision_at_k": hits / k,
        "recall_at_k": (hits / total) if total else 0.0,
    }


def ranker_metrics(df: pd.DataFrame, score_col: str, base_rate: float) -> dict:
    y = df["is_fraud"].to_numpy()
    pos = df.index.get_indexer(_order(df, score_col))
    scores = df[score_col].to_numpy()

    at_k = {}
    for k in K_SWEEP:
        if k > len(df):
            continue
        m = precision_recall_at_k(y, pos, k)
        m["lift_vs_random"] = (m["precision_at_k"] / base_rate) if base_rate > 0 else None
        at_k[str(k)] = m

    try:
        auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None
    except ValueError:
        auc = None
    try:
        ap = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None
    except ValueError:
        ap = None

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "ap_lift_vs_random": (ap / base_rate) if (ap is not None and base_rate > 0) else None,
        "at_k": at_k,
    }


def per_pattern_recall(df: pd.DataFrame, score_col: str) -> dict:
    """For each injected attack type: is it caught, and where does it rank?"""
    y = df["is_fraud"].to_numpy()
    order = _order(df, score_col)
    pos_of = {key: rank for rank, key in enumerate(df.loc[order, "key"], start=1)}
    alerted = set(df.loc[df[score_col] >= ALERT_THRESHOLD, "key"])

    frauds = df[df["is_fraud"] == 1]
    patterns = sorted(set(frauds["fraud_pattern"].unique()) | set(KNOWN_PATTERNS))
    out = {}
    for pattern in patterns:
        if not pattern:
            continue
        grp = frauds[frauds["fraud_pattern"] == pattern]
        n = int(len(grp))
        entry = {"entities": n, "present_in_ground_truth": n > 0}
        if n == 0:
            out[pattern] = entry
            continue
        ranks = np.array([pos_of[k] for k in grp["key"]], dtype=float)
        entry.update(
            {
                "recall_at_alert_cut": float(sum(1 for k in grp["key"] if k in alerted) / n),
                "recall_at_k": {
                    str(k): float((ranks <= k).mean()) for k in K_SWEEP if k <= len(df)
                },
                "median_rank": float(np.median(ranks)),
                "best_rank": int(ranks.min()),
                "worst_rank": int(ranks.max()),
                "median_score": float(grp[score_col].median()),
                "mean_score": float(grp[score_col].mean()),
            }
        )
        out[pattern] = entry
    _ = y  # kept explicit: pattern recall is positional, not threshold-based
    return out


def band_alert_rates(df: pd.DataFrame, score_col: str = "fused_risk_score") -> dict:
    n = len(df)
    total_pos = int(df["is_fraud"].sum())
    out = {}
    for band in ("Medium", "High", "Critical"):
        cut = BAND_THRESHOLDS[band]
        flagged = df[df[score_col] >= cut]
        tp = int(flagged["is_fraud"].sum())
        out[f">= {cut:.2f} ({band}+)"] = {
            "threshold": cut,
            "alerts": int(len(flagged)),
            "alert_rate": (len(flagged) / n) if n else 0.0,
            "true_positives": tp,
            "precision": (tp / len(flagged)) if len(flagged) else 0.0,
            "recall": (tp / total_pos) if total_pos else 0.0,
        }
    return out


def confusion_at(df: pd.DataFrame, threshold: float, score_col: str = "fused_risk_score") -> dict:
    pred = (df[score_col] >= threshold).to_numpy()
    actual = df["is_fraud"].to_numpy().astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / len(df) if len(df) else 0.0,
    }


def band_distribution(df: pd.DataFrame) -> dict:
    if "fused_risk_band" not in df.columns:
        return {}
    counts = df["fused_risk_band"].value_counts()
    out = {}
    for band in BANDS + ["Unknown"]:
        n = int(counts.get(band, 0))
        if not n:
            continue
        grp = df[df["fused_risk_band"] == band]
        out[band] = {
            "entities": n,
            "share": n / len(df),
            "frauds": int(grp["is_fraud"].sum()),
            "fraud_rate": float(grp["is_fraud"].mean()),
        }
    return out


def evaluate_population(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    total_pos = int(df["is_fraud"].sum())
    base_rate = (total_pos / n) if n else 0.0

    per_ranker = {name: ranker_metrics(df, col, base_rate) for name, col in RANKERS.items()}
    # Random selection is reported analytically: the expected precision of a
    # random draw of k is the base rate, exactly, for every k. Sampling one
    # permutation would add noise and report it as a baseline.
    per_ranker["random"] = {
        "roc_auc": 0.5,
        "average_precision": base_rate,
        "ap_lift_vs_random": 1.0,
        "at_k": {
            str(k): {
                "k": min(k, n),
                "true_positives": round(base_rate * min(k, n), 2),
                "precision_at_k": base_rate,
                "recall_at_k": (min(k, n) / n) if n else 0.0,
                "lift_vs_random": 1.0,
            }
            for k in K_SWEEP if k <= n
        },
        "note": "analytic expectation, not a sampled permutation",
    }

    # Lift of the fused ranking against the rules-only ranking at each k.
    lift_vs_rules = {}
    for k in (str(k) for k in K_SWEEP):
        f = per_ranker["fused"]["at_k"].get(k)
        r = per_ranker["rules_only"]["at_k"].get(k)
        if not f or not r:
            continue
        lift_vs_rules[k] = {
            "fused_precision": f["precision_at_k"],
            "rules_precision": r["precision_at_k"],
            "lift": (f["precision_at_k"] / r["precision_at_k"]) if r["precision_at_k"] > 0 else None,
            "extra_frauds_found": f["true_positives"] - r["true_positives"],
        }

    return {
        "population": label,
        "entities": n,
        "frauds": total_pos,
        "base_rate": base_rate,
        "rankers": per_ranker,
        "fused_lift_vs_rules_only": lift_vs_rules,
        "alert_rate_by_band": band_alert_rates(df),
        "confusion_at_shipped_cut": confusion_at(df, ALERT_THRESHOLD),
        "band_distribution": band_distribution(df),
        "per_pattern_recall": {
            "fused": per_pattern_recall(df, "fused_risk_score"),
            "rules_only": per_pattern_recall(df, "rules_score"),
        },
        "comparison_table": comparison_table(per_ranker, base_rate),
    }


def comparison_table(per_ranker: dict, base_rate: float) -> list[dict]:
    rows = []
    for name in ("fused", "ml_only", "rules_only", "random"):
        m = per_ranker.get(name)
        if not m:
            continue
        row = {
            "ranker": name,
            "roc_auc": m["roc_auc"],
            "average_precision": m["average_precision"],
            "ap_lift_vs_random": m["ap_lift_vs_random"],
        }
        for k in K_SWEEP:
            entry = m["at_k"].get(str(k))
            if entry:
                row[f"precision@{k}"] = entry["precision_at_k"]
                row[f"recall@{k}"] = entry["recall_at_k"]
                row[f"lift@{k}"] = entry["lift_vs_random"]
        rows.append(row)
    _ = base_rate
    return rows


def lift_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative precision/recall/lift at every depth, for every ranker."""
    n = len(df)
    y = df["is_fraud"].to_numpy()
    total_pos = int(y.sum())
    base_rate = (total_pos / n) if n else 0.0

    frames = []
    for name, col in RANKERS.items():
        order = df.index.get_indexer(_order(df, col))
        cum_tp = np.cumsum(y[order])
        k = np.arange(1, n + 1)
        frames.append(
            pd.DataFrame(
                {
                    "ranker": name,
                    "k": k,
                    "depth_pct": k / n,
                    "true_positives": cum_tp,
                    "precision_at_k": cum_tp / k,
                    "recall_at_k": (cum_tp / total_pos) if total_pos else 0.0,
                    "lift_vs_random": (cum_tp / k) / base_rate if base_rate > 0 else np.nan,
                }
            )
        )
    k = np.arange(1, n + 1)
    frames.append(
        pd.DataFrame(
            {
                "ranker": "random",
                "k": k,
                "depth_pct": k / n,
                "true_positives": base_rate * k,
                "precision_at_k": base_rate,
                "recall_at_k": k / n,
                "lift_vs_random": 1.0,
            }
        )
    )
    return pd.concat(frames, ignore_index=True)


def leakage_check(df: pd.DataFrame) -> dict:
    """Ground truth must never have reached the weak labels or the features."""
    out = {
        "ground_truth_read_by": "mri.evaluation only",
        "training_labels_source": "V1 rule engine (weak supervision)",
    }
    if "label" in df.columns:
        weak = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        truth = df["is_fraud"].to_numpy()
        agreement = float((weak == truth).mean())
        out["weak_label_vs_ground_truth_agreement"] = agreement
        out["weak_label_positive_rate"] = float(weak.mean())
        out["suspicious"] = bool(agreement > 0.99)
        if out["suspicious"]:
            out["warning"] = (
                "Weak labels agree with ground truth almost perfectly. Either the rule "
                "engine is reading ground truth, or the injected fraud is exactly the "
                "rules. Investigate before trusting any number in this report."
            )
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, diagnostics = build_frame()
    log(f"Joined {diagnostics['ground_truth_entities']} ground-truth entities to scores.")
    if diagnostics["ground_truth_entities_not_scored"]:
        log(
            f"WARNING: {diagnostics['ground_truth_entities_not_scored']} ground-truth entities "
            f"were never scored by the pipeline; they are counted as missed."
        )

    report = {
        "generated_from": {
            "ground_truth": os.path.relpath(GROUND_TRUTH_PATH, BASE_DIR),
            "scores": os.path.relpath(FUSED_PATH, BASE_DIR),
            "rules_baseline": os.path.relpath(RULES_PATH, BASE_DIR),
        },
        "fusion_weights": WEIGHTS,
        "band_thresholds": BAND_THRESHOLDS,
        "shipped_alert_cut": ALERT_THRESHOLD,
        "k_sweep": list(K_SWEEP),
        "join_diagnostics": diagnostics,
        "leakage_check": leakage_check(df),
        "all_entities": evaluate_population(df, "all entities (mostly in-sample)"),
    }

    holdout = df[~df["in_sample"]]
    if len(holdout) >= 10 and holdout["is_fraud"].sum() > 0:
        report["holdout_only"] = evaluate_population(
            holdout, "holdout only (out-of-sample, the honest number)"
        )
    else:
        report["holdout_only"] = {
            "population": "holdout only",
            "entities": int(len(holdout)),
            "frauds": int(holdout["is_fraud"].sum()) if len(holdout) else 0,
            "note": "too small or no positives to evaluate; treat all_entities as in-sample.",
        }

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    lift_curve(df).to_csv(LIFT_CURVE_PATH, index=False)

    log(f"Saved evaluation to: {EVALUATION_PATH}")
    log(f"Saved lift curve to: {LIFT_CURVE_PATH}")
    _print_summary(report)


def _print_summary(report: dict) -> None:
    for section in ("all_entities", "holdout_only"):
        block = report.get(section, {})
        if "rankers" not in block:
            continue
        print(f"\n=== {block['population']} ===")
        print(
            f"{block['entities']} entities, {block['frauds']} frauds "
            f"(base rate {block['base_rate']:.2%})"
        )
        header = f"{'ranker':<12}{'AUC':>8}{'AP':>8}" + "".join(
            f"{'P@'+str(k):>9}" for k in K_SWEEP
        )
        print(header)
        for row in block["comparison_table"]:
            auc = row["roc_auc"]
            ap = row["average_precision"]
            line = (
                f"{row['ranker']:<12}"
                f"{(f'{auc:.3f}' if auc is not None else 'n/a'):>8}"
                f"{(f'{ap:.3f}' if ap is not None else 'n/a'):>8}"
            )
            for k in K_SWEEP:
                v = row.get(f"precision@{k}")
                line += f"{(f'{v:.3f}' if v is not None else '-'):>9}"
            print(line)
        print("\nper-pattern recall (fused):")
        for pattern, stats in block["per_pattern_recall"]["fused"].items():
            if not stats.get("present_in_ground_truth"):
                print(f"  {pattern:<16} not present in ground truth")
                continue
            print(
                f"  {pattern:<16} n={stats['entities']:<5} "
                f"recall@alert={stats['recall_at_alert_cut']:.2%}  "
                f"recall@100={stats['recall_at_k'].get('100', float('nan')):.2%}  "
                f"median rank={stats['median_rank']:.0f}"
            )
        cm = block["confusion_at_shipped_cut"]
        print(
            f"\nconfusion @ {cm['threshold']}: TP={cm['true_positives']} FP={cm['false_positives']} "
            f"FN={cm['false_negatives']} TN={cm['true_negatives']} "
            f"precision={cm['precision']:.3f} recall={cm['recall']:.3f} "
            f"alert rate={cm['alert_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
