import os
import pandas as pd
import numpy as np

from mri.config import ROOT as BASE_DIR  # repository root
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FEATURE_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")
IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")


def log(msg: str):
    print(f"[explain_entities] {msg}")


def zscore_against_group(series: pd.Series, value: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return 0.0
    std = series.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return (value - series.mean()) / std


def classify_reason(feature_name: str, value: float) -> str:
    feature_name = feature_name.lower()

    if "refund" in feature_name:
        return f"Elevated refund behaviour detected ({feature_name}={value:.3f})"
    if "failed" in feature_name:
        return f"Authentication friction observed ({feature_name}={value:.3f})"
    if "device" in feature_name:
        return f"High device concentration detected ({feature_name}={value:.3f})"
    if "region" in feature_name:
        return f"Geographic activity variance observed ({feature_name}={value:.3f})"
    if "spike" in feature_name or "daily" in feature_name:
        return f"Velocity anomaly detected ({feature_name}={value:.3f})"
    if "rating" in feature_name:
        return f"Rating pattern irregularity observed ({feature_name}={value:.3f})"
    if "orders" in feature_name or "order" in feature_name:
        return f"Transaction volume pattern notable ({feature_name}={value:.3f})"
    if "revenue" in feature_name or "spend" in feature_name:
        return f"Monetary activity deviation detected ({feature_name}={value:.3f})"
    if "account_age" in feature_name:
        return f"Account maturity signal notable ({feature_name}={value:.3f})"

    return f"Feature deviation detected ({feature_name}={value:.3f})"


def main():
    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"Missing features file: {FEATURE_PATH}")
    if not os.path.exists(RISK_PATH):
        raise FileNotFoundError(f"Missing risk file: {RISK_PATH}")
    if not os.path.exists(IMPORTANCE_PATH):
        raise FileNotFoundError(f"Missing importance file: {IMPORTANCE_PATH}")

    feature_df = pd.read_csv(FEATURE_PATH)
    risk_df = pd.read_csv(RISK_PATH)
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    if feature_df.empty or risk_df.empty or importance_df.empty:
        raise ValueError("One or more required input files are empty.")

    importance_df = importance_df.sort_values("importance", ascending=False)
    ranked_features = importance_df["feature"].tolist()

    merged = risk_df.merge(
        feature_df,
        on=["entity_id", "entity_type"],
        how="left",
        suffixes=("", "_feature")
    )

    rows = []

    for _, row in merged.iterrows():
        entity_id = row["entity_id"]
        entity_type = row["entity_type"]
        final_risk_score = row.get("final_risk_score", 0)
        risk_band = row.get("risk_band", "Unknown")
        supervised_score = row.get("supervised_score", 0)
        anomaly_score = row.get("anomaly_score", 0)

        peer_group = feature_df[feature_df["entity_type"] == entity_type]

        candidate_reasons = []

        for feature in ranked_features[:20]:
            if feature not in merged.columns:
                continue

            value = pd.to_numeric(pd.Series([row.get(feature, 0)]), errors="coerce").fillna(0).iloc[0]
            group_series = pd.to_numeric(peer_group[feature], errors="coerce") if feature in peer_group.columns else pd.Series(dtype=float)

            z = zscore_against_group(group_series, value)
            weighted_signal = abs(z) * float(
                importance_df.loc[importance_df["feature"] == feature, "importance"].iloc[0]
            )

            candidate_reasons.append({
                "feature": feature,
                "value": value,
                "zscore": z,
                "weighted_signal": weighted_signal,
                "reason_text": classify_reason(feature, value)
            })

        candidate_reasons = sorted(candidate_reasons, key=lambda x: x["weighted_signal"], reverse=True)
        top_reasons = candidate_reasons[:3]

        top_reason_codes = " | ".join([r["feature"] for r in top_reasons])
        top_reason_text = " | ".join([r["reason_text"] for r in top_reasons])

        analyst_summary = (
            f"{entity_type.title()} {entity_id} was classified as {risk_band} risk "
            f"with a fused score of {final_risk_score:.3f}. "
            f"The supervised model scored {supervised_score:.3f} and the anomaly model scored {anomaly_score:.3f}. "
            f"Primary investigation signals include: {top_reason_text}."
        )

        rows.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "risk_band": risk_band,
            "final_risk_score": final_risk_score,
            "supervised_score": supervised_score,
            "anomaly_score": anomaly_score,
            "top_reason_codes": top_reason_codes,
            "top_reason_summary": top_reason_text,
            "analyst_summary": analyst_summary
        })

    explain_df = pd.DataFrame(rows)
    explain_df = explain_df.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    explain_df.to_csv(EXPLAIN_PATH, index=False)

    log(f"Saved explanations to: {EXPLAIN_PATH}")
    print(explain_df.head(10))


if __name__ == "__main__":
    main()