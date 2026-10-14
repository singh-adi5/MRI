import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Hybrid Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

RISK_PATH = os.path.join(OUTPUT_DIR, "final_risk_entities.csv")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")
IMPORTANCE_PATH = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
MODEL_REPORT_PATH = os.path.join(ARTIFACT_DIR, "model_report.json")


@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


risk_df = load_csv(RISK_PATH)
feature_df = load_csv(FEATURE_PATH)
importance_df = load_csv(IMPORTANCE_PATH)
model_report = load_json(MODEL_REPORT_PATH)

st.title("🛡️ Hybrid Risk Intelligence Platform")
st.markdown("V2 Architecture: **Rules + Weak Supervision + Supervised ML + Anomaly Detection + Risk Fusion**")
st.markdown("---")

if risk_df.empty:
    st.error("No risk output found. Run score_entities.py first.")
    st.stop()

# KPI section
total_entities = len(risk_df)
high_risk = (risk_df["risk_band"] == "High").sum() if "risk_band" in risk_df.columns else 0
medium_risk = (risk_df["risk_band"] == "Medium").sum() if "risk_band" in risk_df.columns else 0
avg_score = risk_df["final_risk_score"].mean() if "final_risk_score" in risk_df.columns else 0
seller_count = (risk_df["entity_type"] == "seller").sum() if "entity_type" in risk_df.columns else 0
customer_count = (risk_df["entity_type"] == "customer").sum() if "entity_type" in risk_df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Entities Scored", f"{total_entities:,}")
c2.metric("High Risk", f"{high_risk:,}")
c3.metric("Medium Risk", f"{medium_risk:,}")
c4.metric("Avg Final Risk Score", f"{avg_score:.3f}")
c5.metric("Customers / Sellers", f"{customer_count} / {seller_count}")

st.markdown("---")

# Model report
with st.expander("Model Report"):
    if model_report:
        st.json(model_report)
    else:
        st.info("No model report found.")

# Charts row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Band Distribution")
    if "risk_band" in risk_df.columns:
        band_counts = risk_df["risk_band"].value_counts().reset_index()
        band_counts.columns = ["risk_band", "count"]
        fig_band = px.bar(
            band_counts,
            x="risk_band",
            y="count",
            color="risk_band",
            title="Entities by Risk Band"
        )
        st.plotly_chart(fig_band, use_container_width=True)

with col2:
    st.subheader("Entity Type Split")
    if "entity_type" in risk_df.columns:
        entity_counts = risk_df["entity_type"].value_counts().reset_index()
        entity_counts.columns = ["entity_type", "count"]
        fig_type = px.pie(
            entity_counts,
            names="entity_type",
            values="count",
            title="Customer vs Seller Distribution"
        )
        st.plotly_chart(fig_type, use_container_width=True)

# Charts row 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("Supervised vs Anomaly Score")
    required_cols = {"supervised_score", "anomaly_score", "risk_band", "entity_type"}
    if required_cols.issubset(risk_df.columns):
        fig_scatter = px.scatter(
            risk_df,
            x="supervised_score",
            y="anomaly_score",
            color="risk_band",
            symbol="entity_type",
            hover_data=["entity_id", "entity_type", "final_risk_score"],
            title="Risk Score Fusion Map"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("Top 20 High-Risk Entities")
    if {"entity_id", "entity_type", "final_risk_score"}.issubset(risk_df.columns):
        top20 = risk_df.sort_values("final_risk_score", ascending=False).head(20).copy()
        top20["entity_label"] = top20["entity_type"] + "_" + top20["entity_id"].astype(str)

        fig_top = px.bar(
            top20,
            x="final_risk_score",
            y="entity_label",
            color="entity_type",
            orientation="h",
            title="Top 20 Ranked Risk Entities"
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

# Feature importance
st.markdown("---")
st.subheader("Top Model Features")

if not importance_df.empty and {"feature", "importance"}.issubset(importance_df.columns):
    top_features = importance_df.sort_values("importance", ascending=False).head(15)
    fig_imp = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importances"
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance file not found.")

# Entity drilldown
st.markdown("---")
st.subheader("Entity Drilldown")

entity_type_filter = st.selectbox("Select Entity Type", ["All"] + sorted(risk_df["entity_type"].dropna().unique().tolist()))
risk_band_filter = st.selectbox("Select Risk Band", ["All"] + sorted(risk_df["risk_band"].dropna().unique().tolist()))
search_id = st.text_input("Search Entity ID")

filtered_df = risk_df.copy()

if entity_type_filter != "All":
    filtered_df = filtered_df[filtered_df["entity_type"] == entity_type_filter]

if risk_band_filter != "All":
    filtered_df = filtered_df[filtered_df["risk_band"] == risk_band_filter]

if search_id.strip():
    filtered_df = filtered_df[filtered_df["entity_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)]

st.dataframe(filtered_df, use_container_width=True, height=400)

# Feature lookup section
st.markdown("---")
st.subheader("Feature Lookup for Selected Entity")

if not feature_df.empty and {"entity_id", "entity_type"}.issubset(feature_df.columns):
    selectable = filtered_df.copy()
    selectable["entity_key"] = selectable["entity_type"] + "_" + selectable["entity_id"].astype(str)

    if not selectable.empty:
        selected_key = st.selectbox("Choose an entity", selectable["entity_key"].tolist())

        selected_type, selected_id = selected_key.split("_", 1)

        selected_features = feature_df[
            (feature_df["entity_type"] == selected_type) &
            (feature_df["entity_id"].astype(str) == str(selected_id))
        ]

        if not selected_features.empty:
            st.write("### Entity Feature Snapshot")
            st.dataframe(selected_features, use_container_width=True)
        else:
            st.warning("No feature row found for selected entity.")
    else:
        st.info("No entities available after filtering.")
else:
    st.info("Feature dataset not found.")

st.markdown("---")
st.caption("Architecture: V1 Rules Engine → Weak Labels → Feature Engineering → RandomForest + IsolationForest → Risk Fusion → Investigation Dashboard")