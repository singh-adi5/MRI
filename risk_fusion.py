import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ML_RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
GRAPH_RISK_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
FUSED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v3.csv")


def log(msg: str):
    print(f"[risk_fusion_v3] {msg}")


def normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def assign_fused_band(score: float) -> str:
    if score >= 0.70:
        return "Critical"
    elif score >= 0.50:
        return "High"
    elif score >= 0.25:
        return "Medium"
    return "Low"


def safe_read_csv(path: str, required_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file for {required_name}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{required_name} file is empty: {path}")
    return df


def main():
    log("Loading V2 and V3 outputs...")
    ml_df = safe_read_csv(ML_RISK_PATH, "ML risk output")
    graph_df = safe_read_csv(GRAPH_RISK_PATH, "Graph risk output")
    explain_df = safe_read_csv(EXPLAIN_PATH, "Explanation output")

    # Normalize key columns defensively
    for df in [ml_df, graph_df, explain_df]:
        df["entity_id"] = df["entity_id"].astype(str)
        df["entity_type"] = df["entity_type"].astype(str)

    required_ml = {"entity_id", "entity_type", "supervised_score", "anomaly_score", "final_risk_score", "risk_band"}
    required_graph = {"entity_id", "entity_type", "graph_risk_score", "graph_risk_band"}
    required_explain = {"entity_id", "entity_type", "top_reason_codes", "top_reason_summary", "analyst_summary"}

    if not required_ml.issubset(ml_df.columns):
        raise ValueError(f"ML output missing columns: {required_ml - set(ml_df.columns)}")

    if not required_graph.issubset(graph_df.columns):
        raise ValueError(f"Graph output missing columns: {required_graph - set(graph_df.columns)}")

    if not required_explain.issubset(explain_df.columns):
        raise ValueError(f"Explanation output missing columns: {required_explain - set(explain_df.columns)}")

    log("Merging ML, graph, and explanation layers...")
    fused = ml_df.merge(
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
    ).merge(
        explain_df[
            [
                "entity_id", "entity_type",
                "top_reason_codes", "top_reason_summary", "analyst_summary"
            ]
        ],
        on=["entity_id", "entity_type"],
        how="left"
    )

    # Fill missing graph signals if any entity was absent from graph layer
    graph_fill_cols = [
        "graph_degree", "degree_centrality", "component_size",
        "shared_device_links", "shared_region_links", "transaction_links",
        "graph_risk_score"
    ]
    for col in graph_fill_cols:
        if col in fused.columns:
            fused[col] = pd.to_numeric(fused[col], errors="coerce").fillna(0)

    if "graph_risk_band" in fused.columns:
        fused["graph_risk_band"] = fused["graph_risk_band"].fillna("Low")

    # Ensure numerical types
    fused["supervised_score"] = pd.to_numeric(fused["supervised_score"], errors="coerce").fillna(0)
    fused["anomaly_score"] = pd.to_numeric(fused["anomaly_score"], errors="coerce").fillna(0)
    fused["final_risk_score"] = pd.to_numeric(fused["final_risk_score"], errors="coerce").fillna(0)
    fused["graph_risk_score"] = pd.to_numeric(fused["graph_risk_score"], errors="coerce").fillna(0)

    # Optional normalization for safety
    fused["ml_score_norm"] = normalize(fused["final_risk_score"])
    fused["graph_score_norm"] = normalize(fused["graph_risk_score"])
    fused["anomaly_score_norm"] = normalize(fused["anomaly_score"])

    # Final fusion logic
    # 50% ML fused risk, 30% graph risk, 20% anomaly
    fused["fused_risk_score"] = (
        0.50 * fused["ml_score_norm"] +
        0.30 * fused["graph_score_norm"] +
        0.20 * fused["anomaly_score_norm"]
    )

    fused["fused_risk_score"] = fused["fused_risk_score"].round(6)
    fused["fused_risk_band"] = fused["fused_risk_score"].apply(assign_fused_band)

    # Fused explanation
    fused["fusion_summary"] = fused.apply(
        lambda row: (
            f"{row['entity_type'].title()} {row['entity_id']} has a fused risk score of "
            f"{row['fused_risk_score']:.3f} ({row['fused_risk_band']}). "
            f"ML risk={row['final_risk_score']:.3f}, graph risk={row['graph_risk_score']:.3f}, "
            f"anomaly={row['anomaly_score']:.3f}. "
            f"Primary investigation cues: {row['top_reason_summary'] if pd.notna(row['top_reason_summary']) else 'N/A'}."
        ),
        axis=1
    )

    fused = fused.sort_values("fused_risk_score", ascending=False).reset_index(drop=True)
    fused.to_csv(FUSED_OUTPUT_PATH, index=False)

    log(f"Saved fused V3 risk output to: {FUSED_OUTPUT_PATH}")
    print(fused[
        [
            "entity_id", "entity_type",
            "final_risk_score", "graph_risk_score", "anomaly_score",
            "fused_risk_score", "fused_risk_band"
        ]
    ].head(10))


if __name__ == "__main__":
    main()