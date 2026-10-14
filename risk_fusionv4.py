import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ML_RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
GRAPH_RISK_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")
TEMPORAL_RISK_PATH = os.path.join(OUTPUT_DIR, "temporal_risk_scores.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
FUSED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v4.csv")


def log(msg: str):
    print(f"[risk_fusion_v4] {msg}")


def normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def assign_band(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    elif score >= 0.55:
        return "High"
    elif score >= 0.30:
        return "Medium"
    return "Low"


def safe_read_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def main():
    log("Loading component outputs...")
    ml_df = safe_read_csv(ML_RISK_PATH, "ML risk output")
    graph_df = safe_read_csv(GRAPH_RISK_PATH, "graph risk output")
    temporal_df = safe_read_csv(TEMPORAL_RISK_PATH, "temporal risk output")
    explain_df = safe_read_csv(EXPLAIN_PATH, "explanation output")

    for df in [ml_df, graph_df, temporal_df, explain_df]:
        df["entity_id"] = df["entity_id"].astype(str)
        df["entity_type"] = df["entity_type"].astype(str)

    required_ml = {"entity_id", "entity_type", "supervised_score", "anomaly_score", "final_risk_score", "risk_band"}
    required_graph = {"entity_id", "entity_type", "graph_risk_score", "graph_risk_band"}
    required_temporal = {"entity_id", "entity_type", "temporal_risk_score", "temporal_risk_band"}
    required_explain = {"entity_id", "entity_type", "top_reason_codes", "top_reason_summary", "analyst_summary"}

    if not required_ml.issubset(ml_df.columns):
        raise ValueError(f"ML output missing columns: {required_ml - set(ml_df.columns)}")
    if not required_graph.issubset(graph_df.columns):
        raise ValueError(f"Graph output missing columns: {required_graph - set(graph_df.columns)}")
    if not required_temporal.issubset(temporal_df.columns):
        raise ValueError(f"Temporal output missing columns: {required_temporal - set(temporal_df.columns)}")
    if not required_explain.issubset(explain_df.columns):
        raise ValueError(f"Explanation output missing columns: {required_explain - set(explain_df.columns)}")

    log("Merging ML, graph, temporal, and explanation layers...")
    fused = (
        ml_df.merge(
            graph_df[
                [
                    "entity_id", "entity_type",
                    "graph_degree", "degree_centrality", "component_size",
                    "shared_device_links", "shared_region_links",
                    "transaction_links", "graph_risk_score", "graph_risk_band"
                ]
            ],
            on=["entity_id", "entity_type"],
            how="left"
        )
        .merge(
            temporal_df[
                [
                    "entity_id", "entity_type",
                    "temporal_risk_score", "temporal_risk_band",
                    "order_velocity_per_hour", "order_burst_ratio",
                    "refund_velocity_per_hour", "login_velocity_per_hour",
                    "failed_attempt_density", "seller_order_velocity_per_hour",
                    "seller_order_burst_ratio"
                ]
            ],
            on=["entity_id", "entity_type"],
            how="left"
        )
        .merge(
            explain_df[
                [
                    "entity_id", "entity_type",
                    "top_reason_codes", "top_reason_summary", "analyst_summary"
                ]
            ],
            on=["entity_id", "entity_type"],
            how="left"
        )
    )

    numeric_fill_cols = [
        "supervised_score", "anomaly_score", "final_risk_score",
        "graph_degree", "degree_centrality", "component_size",
        "shared_device_links", "shared_region_links", "transaction_links",
        "graph_risk_score",
        "temporal_risk_score", "order_velocity_per_hour", "order_burst_ratio",
        "refund_velocity_per_hour", "login_velocity_per_hour",
        "failed_attempt_density", "seller_order_velocity_per_hour",
        "seller_order_burst_ratio"
    ]

    for col in numeric_fill_cols:
        if col in fused.columns:
            fused[col] = pd.to_numeric(fused[col], errors="coerce").fillna(0)

    for col in ["graph_risk_band", "temporal_risk_band", "risk_band"]:
        if col in fused.columns:
            fused[col] = fused[col].fillna("Low")

    # Normalize component scores
    fused["ml_score_norm"] = normalize(fused["final_risk_score"])
    fused["graph_score_norm"] = normalize(fused["graph_risk_score"])
    fused["temporal_score_norm"] = normalize(fused["temporal_risk_score"])
    fused["anomaly_score_norm"] = normalize(fused["anomaly_score"])

    # Final multi-layer fusion
    # 40% ML risk, 25% Graph risk, 20% Temporal risk, 15% anomaly
    fused["fused_risk_score"] = (
        0.40 * fused["ml_score_norm"] +
        0.25 * fused["graph_score_norm"] +
        0.20 * fused["temporal_score_norm"] +
        0.15 * fused["anomaly_score_norm"]
    ).round(6)

    fused["fused_risk_band"] = fused["fused_risk_score"].apply(assign_band)

    fused["fusion_summary"] = fused.apply(
        lambda row: (
            f"{row['entity_type'].title()} {row['entity_id']} has a fused risk score of "
            f"{row['fused_risk_score']:.3f} ({row['fused_risk_band']}). "
            f"ML={row['final_risk_score']:.3f}, Graph={row['graph_risk_score']:.3f}, "
            f"Temporal={row['temporal_risk_score']:.3f}, Anomaly={row['anomaly_score']:.3f}. "
            f"Primary cues: {row['top_reason_summary'] if pd.notna(row['top_reason_summary']) else 'N/A'}."
        ),
        axis=1
    )

    fused = fused.sort_values("fused_risk_score", ascending=False).reset_index(drop=True)
    fused.to_csv(FUSED_OUTPUT_PATH, index=False)

    log(f"Saved fused V4 output to: {FUSED_OUTPUT_PATH}")
    print(
        fused[
            [
                "entity_id", "entity_type",
                "final_risk_score", "graph_risk_score", "temporal_risk_score",
                "anomaly_score", "fused_risk_score", "fused_risk_band"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()