import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="V5 Analyst Investigation Workbench",
    page_icon="🛡️",
    layout="wide"
)

from mri.config import ROOT as BASE_DIR  # repository root
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

FUSED_PATH = os.path.join(OUTPUT_DIR, "final_fused_risk_v4.csv")
EXPLAIN_PATH = os.path.join(OUTPUT_DIR, "entity_explanations.csv")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")
TEMPORAL_PATH = os.path.join(OUTPUT_DIR, "temporal_risk_scores.csv")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def build_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and {"entity_type", "entity_id"}.issubset(df.columns):
        df = df.copy()
        df["entity_key"] = df["entity_type"].astype(str) + "_" + df["entity_id"].astype(str)
    return df


fused_df = build_entity_key(load_csv(FUSED_PATH))
explain_df = build_entity_key(load_csv(EXPLAIN_PATH))
graph_df = build_entity_key(load_csv(GRAPH_PATH))
temporal_df = build_entity_key(load_csv(TEMPORAL_PATH))
feature_df = build_entity_key(load_csv(FEATURE_PATH))

st.title("🛡️ V5 Analyst Investigation Workbench")
st.markdown("**Multi-Layer Risk Intelligence Console** | Rules + ML + Anomaly + Graph + Temporal")
st.markdown("---")

if fused_df.empty:
    st.error("No fused risk file found. Run `risk_fusion_v4.py` first.")
    st.stop()

# =========================
# KPI SECTION
# =========================
total_entities = len(fused_df)
critical_cases = (fused_df["fused_risk_band"] == "Critical").sum() if "fused_risk_band" in fused_df.columns else 0
high_cases = (fused_df["fused_risk_band"] == "High").sum() if "fused_risk_band" in fused_df.columns else 0
avg_fused_score = fused_df["fused_risk_score"].mean() if "fused_risk_score" in fused_df.columns else 0
seller_cases = (fused_df["entity_type"] == "seller").sum() if "entity_type" in fused_df.columns else 0
customer_cases = (fused_df["entity_type"] == "customer").sum() if "entity_type" in fused_df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Entities in Queue", f"{total_entities:,}")
k2.metric("Critical Cases", f"{critical_cases:,}")
k3.metric("High Cases", f"{high_cases:,}")
k4.metric("Average Fused Score", f"{avg_fused_score:.3f}")
k5.metric("Customers / Sellers", f"{customer_cases} / {seller_cases}")

st.markdown("---")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Investigation Filters")

entity_type_filter = st.sidebar.multiselect(
    "Entity Type",
    options=sorted(fused_df["entity_type"].dropna().unique().tolist()),
    default=sorted(fused_df["entity_type"].dropna().unique().tolist())
)

risk_band_filter = st.sidebar.multiselect(
    "Fused Risk Band",
    options=sorted(fused_df["fused_risk_band"].dropna().unique().tolist()),
    default=sorted(fused_df["fused_risk_band"].dropna().unique().tolist())
)

min_score = float(fused_df["fused_risk_score"].min())
max_score = float(fused_df["fused_risk_score"].max())

score_range = st.sidebar.slider(
    "Fused Risk Score Range",
    min_value=float(round(min_score, 3)),
    max_value=float(round(max_score, 3)),
    value=(float(round(min_score, 3)), float(round(max_score, 3)))
)

search_term = st.sidebar.text_input("Search Entity ID / Key")
top_n = st.sidebar.slider("Queue Size", min_value=10, max_value=200, value=50, step=10)

filtered_df = fused_df.copy()

if entity_type_filter:
    filtered_df = filtered_df[filtered_df["entity_type"].isin(entity_type_filter)]

if risk_band_filter:
    filtered_df = filtered_df[filtered_df["fused_risk_band"].isin(risk_band_filter)]

filtered_df = filtered_df[
    (filtered_df["fused_risk_score"] >= score_range[0]) &
    (filtered_df["fused_risk_score"] <= score_range[1])
]

if search_term.strip():
    filtered_df = filtered_df[
        filtered_df["entity_key"].str.contains(search_term.strip(), case=False, na=False) |
        filtered_df["entity_id"].astype(str).str.contains(search_term.strip(), case=False, na=False)
    ]

filtered_df = filtered_df.sort_values("fused_risk_score", ascending=False).head(top_n)

# =========================
# QUEUE VISUALS
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Queue by Fused Risk Band")
    if "fused_risk_band" in filtered_df.columns:
        band_counts = filtered_df["fused_risk_band"].value_counts().reset_index()
        band_counts.columns = ["fused_risk_band", "count"]
        fig_band = px.bar(
            band_counts,
            x="fused_risk_band",
            y="count",
            color="fused_risk_band",
            title="Investigation Queue Distribution"
        )
        st.plotly_chart(fig_band, use_container_width=True)

with c2:
    st.subheader("Risk Intelligence Scatter")
    required_cols = {"graph_risk_score", "temporal_risk_score", "fused_risk_band", "entity_type"}
    if required_cols.issubset(filtered_df.columns):
        fig_scatter = px.scatter(
            filtered_df,
            x="graph_risk_score",
            y="temporal_risk_score",
            size="fused_risk_score",
            color="fused_risk_band",
            symbol="entity_type",
            hover_data=["entity_id", "entity_type", "fused_risk_score"],
            title="Graph vs Temporal Intelligence"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# =========================
# INVESTIGATION QUEUE TABLE
# =========================
st.subheader("Investigation Queue")

queue_cols = [
    "entity_key", "entity_type", "fused_risk_band", "fused_risk_score",
    "final_risk_score", "graph_risk_score", "temporal_risk_score", "anomaly_score"
]
queue_cols = [c for c in queue_cols if c in filtered_df.columns]

st.dataframe(filtered_df[queue_cols], use_container_width=True, height=350)

if filtered_df.empty:
    st.warning("No entities match the current filters.")
    st.stop()

st.markdown("---")

# =========================
# ENTITY SELECTION
# =========================
st.subheader("Case Investigation Panel")

selected_entity = st.selectbox(
    "Select Entity from Queue",
    filtered_df["entity_key"].tolist()
)

selected_row = filtered_df[filtered_df["entity_key"] == selected_entity].iloc[0]

left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Core Case Summary")
    st.write(f"**Entity:** {selected_row['entity_key']}")
    st.write(f"**Entity Type:** {selected_row['entity_type']}")
    st.write(f"**Fused Risk Band:** {selected_row['fused_risk_band']}")
    st.write(f"**Fused Risk Score:** {selected_row['fused_risk_score']:.3f}")

    if "fusion_summary" in selected_row:
        st.info(selected_row["fusion_summary"])

    explain_match = explain_df[explain_df["entity_key"] == selected_entity]
    if not explain_match.empty:
        explain_row = explain_match.iloc[0]
        st.markdown("### Analyst Narrative")
        if "analyst_summary" in explain_row:
            st.write(explain_row["analyst_summary"])
        if "top_reason_codes" in explain_row:
            st.write(f"**Reason Codes:** {explain_row['top_reason_codes']}")
        if "top_reason_summary" in explain_row:
            st.write(f"**Reason Summary:** {explain_row['top_reason_summary']}")

with right:
    st.markdown("### Score Breakdown")
    score_breakdown = pd.DataFrame({
        "signal": ["ML", "Graph", "Temporal", "Anomaly"],
        "score": [
            float(selected_row.get("final_risk_score", 0)),
            float(selected_row.get("graph_risk_score", 0)),
            float(selected_row.get("temporal_risk_score", 0)),
            float(selected_row.get("anomaly_score", 0))
        ]
    })
    fig_breakdown = px.bar(
        score_breakdown,
        x="signal",
        y="score",
        color="signal",
        title="Signal Contribution View"
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

st.markdown("---")

# =========================
# ENTITY EVIDENCE PANELS
# =========================
e1, e2 = st.columns(2)

with e1:
    st.markdown("### Graph Intelligence Evidence")
    graph_match = graph_df[graph_df["entity_key"] == selected_entity]
    if not graph_match.empty:
        graph_row = graph_match.iloc[0]
        graph_evidence = graph_row.to_frame(name="value")
        st.dataframe(graph_evidence, use_container_width=True)
    else:
        st.info("No graph evidence available.")

with e2:
    st.markdown("### Temporal Intelligence Evidence")
    temporal_match = temporal_df[temporal_df["entity_key"] == selected_entity]
    if not temporal_match.empty:
        temporal_row = temporal_match.iloc[0]
        temporal_evidence = temporal_row.to_frame(name="value")
        st.dataframe(temporal_evidence, use_container_width=True)
    else:
        st.info("No temporal evidence available.")

st.markdown("---")

# =========================
# FEATURE SNAPSHOT
# =========================
st.markdown("### Engineered Feature Snapshot")
feature_match = feature_df[feature_df["entity_key"] == selected_entity]
if not feature_match.empty:
    st.dataframe(feature_match, use_container_width=True)
else:
    st.info("No engineered feature row available.")

st.markdown("---")

# =========================
# CASE ACTION PANEL
# =========================
st.markdown("### Analyst Action Panel")

a1, a2, a3 = st.columns(3)
with a1:
    case_status = st.selectbox(
        "Case Status",
        ["New", "In Review", "Escalated", "Monitor", "Closed"]
    )
with a2:
    recommended_action = st.selectbox(
        "Recommended Action",
        ["No Action", "Monitor", "Request Manual Review", "Restrict", "Escalate"]
    )
with a3:
    priority = st.selectbox(
        "Priority",
        ["P3", "P2", "P1", "Critical"]
    )

analyst_notes = st.text_area(
    "Analyst Notes",
    value=f"Review initiated for {selected_entity}.",
    height=140
)

case_payload = {
    "entity_key": selected_entity,
    "case_status": case_status,
    "recommended_action": recommended_action,
    "priority": priority,
    "analyst_notes": analyst_notes
}

st.markdown("### Review Payload Preview")
st.code(json.dumps(case_payload, indent=2), language="json")

st.markdown("---")
st.caption("V5 Analyst Investigation Workbench | Multi-Layer Fraud Intelligence Console")