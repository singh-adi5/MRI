import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os

st.set_page_config(
    page_title="Marketplace Risk Command Center",
    layout="wide",
    page_icon="🛡️"
)

from mri.config import ROOT as BASE_DIR  # repository root
DB_PATH = os.path.join(BASE_DIR, "data", "marketplace.db")
FLAGGED_CSV = os.path.join(BASE_DIR, "outputs", "flagged_entities.csv")


@st.cache_data
def load_sql_data(query):
    """Run read-only SQL query and return DataFrame."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Database query error: {e}")
        return pd.DataFrame()


@st.cache_data
def load_flagged_entities():
    """Load flagged entities CSV."""
    try:
        if os.path.exists(FLAGGED_CSV):
            return pd.read_csv(FLAGGED_CSV)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"CSV load error: {e}")
        return pd.DataFrame()


def safe_get_count(query):
    """Safely fetch count from SQL query."""
    df = load_sql_data(query)
    if not df.empty and "count" in df.columns:
        return int(df.iloc[0]["count"])
    return 0


st.title("🛡️ Marketplace Risk & Fraud Command Center")
st.markdown("Operational Risk Monitoring Pipeline | **Environment:** Staging | **Status:** Active")
st.markdown("---")

# KPI SECTION
col1, col2, col3, col4 = st.columns(4)

total_orders = safe_get_count("SELECT COUNT(*) AS count FROM orders")
total_refunds = safe_get_count("SELECT COUNT(*) AS count FROM refunds")
failed_logins = safe_get_count(
    "SELECT COUNT(*) AS count FROM login_activity WHERE failed_attempts > 0"
)

flagged_df = load_flagged_entities()
total_flagged = len(flagged_df) if not flagged_df.empty else 0

col1.metric("Total Transactions Validated", f"{total_orders:,}")
col2.metric("Refund Events", f"{total_refunds:,}")
col3.metric("Failed Authentication Attempts", f"{failed_logins:,}")
col4.metric("High-Risk Entities Flagged", f"{total_flagged:,}")

st.markdown("---")

# VISUALS
colA, colB = st.columns(2)

with colA:
    st.subheader("Login Anomalies")
    logins_df = load_sql_data("""
        SELECT login_date, failed_attempts
        FROM login_activity
    """)

    if not logins_df.empty:
        logins_df["login_date"] = pd.to_datetime(logins_df["login_date"], errors="coerce")
        logins_df = logins_df.dropna(subset=["login_date"])
        logins_df["date"] = logins_df["login_date"].dt.date
        logins_df["login_status"] = logins_df["failed_attempts"].apply(
            lambda x: "Failed Attempt" if x > 0 else "Successful Login"
        )

        daily_logins = (
            logins_df.groupby(["date", "login_status"])
            .size()
            .reset_index(name="count")
        )

        fig_logins = px.bar(
            daily_logins,
            x="date",
            y="count",
            color="login_status",
            title="Daily Login Activity by Status",
            barmode="group"
        )
        st.plotly_chart(fig_logins, use_container_width=True)
    else:
        st.warning("No login activity data available.")

with colB:
    st.subheader("High-Risk Entities Pipeline")
    if not flagged_df.empty:
        if "risk_score" in flagged_df.columns:
            fig_risk = px.histogram(
                flagged_df,
                x="risk_score",
                nbins=20,
                title="Distribution of Risk Scores",
                color_discrete_sequence=["#ef553b"]
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("The flagged_entities.csv file does not contain a risk_score column.")
    else:
        st.warning("No flagged entities found. Run risk_scoring.py first.")

st.markdown("---")

# EXTRA CHARTS
colC, colD = st.columns(2)

with colC:
    st.subheader("Flagged Entities by Type")
    if not flagged_df.empty and "entity_type" in flagged_df.columns:
        type_counts = flagged_df["entity_type"].value_counts().reset_index()
        type_counts.columns = ["entity_type", "count"]

        fig_type = px.pie(
            type_counts,
            names="entity_type",
            values="count",
            title="Customer vs Seller Risk Distribution"
        )
        st.plotly_chart(fig_type, use_container_width=True)

with colD:
    st.subheader("Top Risk Reasons")
    if not flagged_df.empty and "reasons" in flagged_df.columns:
        reason_series = flagged_df["reasons"].dropna().str.split("; ").explode()
        reason_counts = reason_series.value_counts().head(10).reset_index()
        reason_counts.columns = ["reason", "count"]

        fig_reason = px.bar(
            reason_counts,
            x="count",
            y="reason",
            orientation="h",
            title="Top 10 Risk Triggers"
        )
        st.plotly_chart(fig_reason, use_container_width=True)

# TABLE
st.subheader("🚨 Flagged Entity Matrix")
if not flagged_df.empty:
    st.dataframe(flagged_df, use_container_width=True)
else:
    st.info("Awaiting pipeline execution to populate risk matrix.")

st.caption("Architecture: Python | SQLite | Pandas | Streamlit")