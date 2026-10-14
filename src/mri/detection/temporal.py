import os
import sqlite3
import pandas as pd
import numpy as np

from mri.config import ROOT as BASE_DIR  # repository root
DB_PATH = os.path.join(BASE_DIR, "data", "marketplace.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "temporal_risk_scores.csv")


def log(msg: str):
    print(f"[temporal_intelligence] {msg}")


def load_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def assign_band(score: float) -> str:
    if score >= 0.60:
        return "High"
    elif score >= 0.30:
        return "Medium"
    return "Low"


def safe_hours_between(min_ts, max_ts):
    if pd.isna(min_ts) or pd.isna(max_ts):
        return 0.0
    delta_hours = (max_ts - min_ts).total_seconds() / 3600.0
    return max(delta_hours, 0.0)


def compute_order_temporal_features(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=[
            "entity_id", "entity_type",
            "order_count", "order_active_hours", "order_velocity_per_hour",
            "max_orders_in_day", "avg_orders_per_day", "order_burst_ratio"
        ])

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders = orders.dropna(subset=["order_date"]).copy()

    customer_order_stats = (
        orders.groupby("customer_id")
        .agg(
            order_count=("order_id", "count"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max")
        )
        .reset_index()
    )

    customer_order_stats["order_active_hours"] = customer_order_stats.apply(
        lambda row: safe_hours_between(row["first_order"], row["last_order"]),
        axis=1
    )
    customer_order_stats["order_velocity_per_hour"] = (
        customer_order_stats["order_count"] /
        customer_order_stats["order_active_hours"].replace(0, 1)
    )

    orders["order_day"] = orders["order_date"].dt.date
    customer_daily = (
        orders.groupby(["customer_id", "order_day"])
        .size()
        .reset_index(name="daily_orders")
    )

    customer_burst = (
        customer_daily.groupby("customer_id")
        .agg(
            max_orders_in_day=("daily_orders", "max"),
            avg_orders_per_day=("daily_orders", "mean")
        )
        .reset_index()
    )
    customer_burst["order_burst_ratio"] = (
        customer_burst["max_orders_in_day"] /
        customer_burst["avg_orders_per_day"].replace(0, 1)
    )

    customer_temporal = customer_order_stats.merge(customer_burst, on="customer_id", how="left")
    customer_temporal["entity_id"] = customer_temporal["customer_id"].astype(str)
    customer_temporal["entity_type"] = "customer"

    return customer_temporal[[
        "entity_id", "entity_type",
        "order_count", "order_active_hours", "order_velocity_per_hour",
        "max_orders_in_day", "avg_orders_per_day", "order_burst_ratio"
    ]]


def compute_refund_temporal_features(refunds: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if refunds.empty or orders.empty:
        return pd.DataFrame(columns=[
            "entity_id", "entity_type",
            "refund_count", "refund_active_hours", "refund_velocity_per_hour"
        ])

    refunds["refund_date"] = pd.to_datetime(refunds["refund_date"], errors="coerce")
    refunds = refunds.dropna(subset=["refund_date"]).copy()

    refund_join = refunds.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="left"
    )

    customer_refund_stats = (
        refund_join.groupby("customer_id")
        .agg(
            refund_count=("refund_id", "count"),
            first_refund=("refund_date", "min"),
            last_refund=("refund_date", "max")
        )
        .reset_index()
    )

    customer_refund_stats["refund_active_hours"] = customer_refund_stats.apply(
        lambda row: safe_hours_between(row["first_refund"], row["last_refund"]),
        axis=1
    )
    customer_refund_stats["refund_velocity_per_hour"] = (
        customer_refund_stats["refund_count"] /
        customer_refund_stats["refund_active_hours"].replace(0, 1)
    )

    customer_refund_stats["entity_id"] = customer_refund_stats["customer_id"].astype(str)
    customer_refund_stats["entity_type"] = "customer"

    return customer_refund_stats[[
        "entity_id", "entity_type",
        "refund_count", "refund_active_hours", "refund_velocity_per_hour"
    ]]


def compute_login_temporal_features(login: pd.DataFrame) -> pd.DataFrame:
    if login.empty:
        return pd.DataFrame(columns=[
            "entity_id", "entity_type",
            "login_count", "login_active_hours", "login_velocity_per_hour",
            "max_failed_attempts", "failed_attempt_density"
        ])

    login["login_date"] = pd.to_datetime(login["login_date"], errors="coerce")
    login = login.dropna(subset=["login_date"]).copy()

    login["user_id"] = login["user_id"].astype(str)
    login["entity_type"] = login["user_id"].str[0].map({"C": "customer", "S": "seller"})
    login["entity_id"] = login["user_id"].str[1:]

    login_stats = (
        login.groupby(["entity_id", "entity_type"])
        .agg(
            login_count=("user_id", "count"),
            first_login=("login_date", "min"),
            last_login=("login_date", "max"),
            max_failed_attempts=("failed_attempts", "max"),
            total_failed_attempts=("failed_attempts", "sum"),
            unique_devices=("device_id", "nunique"),
            unique_regions=("ip_region", "nunique")
        )
        .reset_index()
    )

    login_stats["login_active_hours"] = login_stats.apply(
        lambda row: safe_hours_between(row["first_login"], row["last_login"]),
        axis=1
    )
    login_stats["login_velocity_per_hour"] = (
        login_stats["login_count"] /
        login_stats["login_active_hours"].replace(0, 1)
    )
    login_stats["failed_attempt_density"] = (
        login_stats["total_failed_attempts"] /
        login_stats["login_count"].replace(0, 1)
    )

    return login_stats[[
        "entity_id", "entity_type",
        "login_count", "login_active_hours", "login_velocity_per_hour",
        "max_failed_attempts", "failed_attempt_density",
        "unique_devices", "unique_regions"
    ]]


def compute_seller_order_temporal_features(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=[
            "entity_id", "entity_type",
            "seller_order_count", "seller_active_hours",
            "seller_order_velocity_per_hour", "seller_max_orders_in_day",
            "seller_avg_orders_per_day", "seller_order_burst_ratio"
        ])

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders = orders.dropna(subset=["order_date"]).copy()

    seller_stats = (
        orders.groupby("seller_id")
        .agg(
            seller_order_count=("order_id", "count"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max")
        )
        .reset_index()
    )

    seller_stats["seller_active_hours"] = seller_stats.apply(
        lambda row: safe_hours_between(row["first_order"], row["last_order"]),
        axis=1
    )
    seller_stats["seller_order_velocity_per_hour"] = (
        seller_stats["seller_order_count"] /
        seller_stats["seller_active_hours"].replace(0, 1)
    )

    orders["order_day"] = orders["order_date"].dt.date
    seller_daily = (
        orders.groupby(["seller_id", "order_day"])
        .size()
        .reset_index(name="daily_orders")
    )

    seller_burst = (
        seller_daily.groupby("seller_id")
        .agg(
            seller_max_orders_in_day=("daily_orders", "max"),
            seller_avg_orders_per_day=("daily_orders", "mean")
        )
        .reset_index()
    )

    seller_burst["seller_order_burst_ratio"] = (
        seller_burst["seller_max_orders_in_day"] /
        seller_burst["seller_avg_orders_per_day"].replace(0, 1)
    )

    seller_temporal = seller_stats.merge(seller_burst, on="seller_id", how="left")
    seller_temporal["entity_id"] = seller_temporal["seller_id"].astype(str)
    seller_temporal["entity_type"] = "seller"

    return seller_temporal[[
        "entity_id", "entity_type",
        "seller_order_count", "seller_active_hours",
        "seller_order_velocity_per_hour", "seller_max_orders_in_day",
        "seller_avg_orders_per_day", "seller_order_burst_ratio"
    ]]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log("Loading source tables...")
    orders = load_sql("SELECT order_id, customer_id, seller_id, order_date FROM orders")
    refunds = load_sql("SELECT refund_id, order_id, refund_date FROM refunds")
    login = load_sql("SELECT user_id, login_date, failed_attempts, device_id, ip_region FROM login_activity")

    log("Computing customer order temporal features...")
    customer_order_temporal = compute_order_temporal_features(orders)

    log("Computing customer refund temporal features...")
    customer_refund_temporal = compute_refund_temporal_features(refunds, orders)

    log("Computing login temporal features...")
    login_temporal = compute_login_temporal_features(login)

    log("Computing seller order temporal features...")
    seller_order_temporal = compute_seller_order_temporal_features(orders)

    log("Merging temporal layers...")
    customer_temporal = customer_order_temporal.merge(
        customer_refund_temporal,
        on=["entity_id", "entity_type"],
        how="outer"
    ).merge(
        login_temporal[login_temporal["entity_type"] == "customer"],
        on=["entity_id", "entity_type"],
        how="outer"
    )

    seller_temporal = seller_order_temporal.merge(
        login_temporal[login_temporal["entity_type"] == "seller"],
        on=["entity_id", "entity_type"],
        how="outer"
    )

    temporal_df = pd.concat([customer_temporal, seller_temporal], ignore_index=True, sort=False)
    temporal_df = temporal_df.fillna(0)

    # Normalize temporal signals
    signal_cols = [
        "order_velocity_per_hour",
        "order_burst_ratio",
        "refund_velocity_per_hour",
        "login_velocity_per_hour",
        "failed_attempt_density",
        "unique_devices",
        "unique_regions",
        "seller_order_velocity_per_hour",
        "seller_order_burst_ratio"
    ]

    for col in signal_cols:
        if col not in temporal_df.columns:
            temporal_df[col] = 0

    temporal_df["norm_order_velocity"] = normalize(pd.to_numeric(temporal_df["order_velocity_per_hour"], errors="coerce").fillna(0))
    temporal_df["norm_order_burst"] = normalize(pd.to_numeric(temporal_df["order_burst_ratio"], errors="coerce").fillna(0))
    temporal_df["norm_refund_velocity"] = normalize(pd.to_numeric(temporal_df["refund_velocity_per_hour"], errors="coerce").fillna(0))
    temporal_df["norm_login_velocity"] = normalize(pd.to_numeric(temporal_df["login_velocity_per_hour"], errors="coerce").fillna(0))
    temporal_df["norm_failed_density"] = normalize(pd.to_numeric(temporal_df["failed_attempt_density"], errors="coerce").fillna(0))
    temporal_df["norm_unique_devices"] = normalize(pd.to_numeric(temporal_df["unique_devices"], errors="coerce").fillna(0))
    temporal_df["norm_unique_regions"] = normalize(pd.to_numeric(temporal_df["unique_regions"], errors="coerce").fillna(0))
    temporal_df["norm_seller_velocity"] = normalize(pd.to_numeric(temporal_df["seller_order_velocity_per_hour"], errors="coerce").fillna(0))
    temporal_df["norm_seller_burst"] = normalize(pd.to_numeric(temporal_df["seller_order_burst_ratio"], errors="coerce").fillna(0))

    # Fused temporal risk score
    temporal_df["temporal_risk_score"] = (
        0.15 * temporal_df["norm_order_velocity"] +
        0.15 * temporal_df["norm_order_burst"] +
        0.15 * temporal_df["norm_refund_velocity"] +
        0.15 * temporal_df["norm_login_velocity"] +
        0.10 * temporal_df["norm_failed_density"] +
        0.10 * temporal_df["norm_unique_devices"] +
        0.05 * temporal_df["norm_unique_regions"] +
        0.075 * temporal_df["norm_seller_velocity"] +
        0.075 * temporal_df["norm_seller_burst"]
    )

    temporal_df["temporal_risk_score"] = temporal_df["temporal_risk_score"].round(6)
    temporal_df["temporal_risk_band"] = temporal_df["temporal_risk_score"].apply(assign_band)

    temporal_df = temporal_df.sort_values("temporal_risk_score", ascending=False).reset_index(drop=True)
    temporal_df.to_csv(OUTPUT_PATH, index=False)

    log(f"Saved temporal risk scores to: {OUTPUT_PATH}")
    print(temporal_df[[
        "entity_id", "entity_type", "temporal_risk_score", "temporal_risk_band"
    ]].head(10))


if __name__ == "__main__":
    main()