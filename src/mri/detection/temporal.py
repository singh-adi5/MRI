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


# ---------------------------------------------------------------------------
# Velocity measurability thresholds
# ---------------------------------------------------------------------------
# A rate needs an interval. An entity with a single event has an observation
# window of zero hours, and dividing by a 1-hour floor (the previous
# ``.replace(0, 1)``) handed it the MAXIMUM possible velocity -- a sign
# inversion that ranked one-order accounts above genuine bursts.
#
# Rules now applied to every velocity feature:
#   * fewer than MIN_EVENTS_FOR_VELOCITY timestamps -> velocity is NaN
#     (not measurable; recorded as such, and contributes nothing to the score)
#   * at least MIN_EVENTS_FOR_VELOCITY timestamps -> divide by the observed
#     window floored at MIN_OBSERVATION_HOURS, so simultaneous/near-simultaneous
#     events stay a bounded high value instead of diverging.
MIN_EVENTS_FOR_VELOCITY = 2
MIN_OBSERVATION_HOURS = 1.0


def velocity_per_hour(event_count: pd.Series, active_hours: pd.Series) -> pd.Series:
    """Events per hour, or NaN where the observation window cannot support a rate."""
    count = pd.to_numeric(event_count, errors="coerce")
    hours = pd.to_numeric(active_hours, errors="coerce")
    denominator = hours.clip(lower=MIN_OBSERVATION_HOURS)
    velocity = count / denominator
    measurable = count >= MIN_EVENTS_FOR_VELOCITY
    return velocity.where(measurable, other=np.nan)


def hours_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Vectorised observation window in hours (negative/NaT -> 0.0)."""
    delta = (pd.to_datetime(end) - pd.to_datetime(start)).dt.total_seconds() / 3600.0
    return delta.fillna(0.0).clip(lower=0.0)


def _check_query(query: str) -> str:
    """Accept only a single literal SELECT. Queries in this module are constants;
    this guard keeps that true if a caller is ever changed."""
    q = query.strip().rstrip(";")
    if ";" in q or not q.upper().startswith("SELECT"):
        raise ValueError("only a single SELECT statement is allowed")
    return q


def load_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(_check_query(query), conn)


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

    customer_order_stats["order_active_hours"] = hours_between(
        customer_order_stats["first_order"], customer_order_stats["last_order"]
    )
    customer_order_stats["order_velocity_per_hour"] = velocity_per_hour(
        customer_order_stats["order_count"], customer_order_stats["order_active_hours"]
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
    # avg_orders_per_day is a mean over observed days, so it is > 0 by
    # construction; NaN (rather than 1) is the honest result if it ever is not.
    customer_burst["order_burst_ratio"] = (
        customer_burst["max_orders_in_day"] /
        customer_burst["avg_orders_per_day"].replace(0, np.nan)
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

    customer_refund_stats["refund_active_hours"] = hours_between(
        customer_refund_stats["first_refund"], customer_refund_stats["last_refund"]
    )
    customer_refund_stats["refund_velocity_per_hour"] = velocity_per_hour(
        customer_refund_stats["refund_count"], customer_refund_stats["refund_active_hours"]
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

    login_stats["login_active_hours"] = hours_between(
        login_stats["first_login"], login_stats["last_login"]
    )
    login_stats["login_velocity_per_hour"] = velocity_per_hour(
        login_stats["login_count"], login_stats["login_active_hours"]
    )
    # Density is a per-event average, not a rate: one login is enough to
    # measure it, so only a genuinely absent denominator yields NaN.
    login_stats["failed_attempt_density"] = (
        login_stats["total_failed_attempts"] /
        login_stats["login_count"].replace(0, np.nan)
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

    seller_stats["seller_active_hours"] = hours_between(
        seller_stats["first_order"], seller_stats["last_order"]
    )
    seller_stats["seller_order_velocity_per_hour"] = velocity_per_hour(
        seller_stats["seller_order_count"], seller_stats["seller_active_hours"]
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
        seller_burst["seller_avg_orders_per_day"].replace(0, np.nan)
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

    # Signals whose NaN means "not measurable" must keep their NaN. Counts and
    # window lengths, by contrast, are genuinely zero for an entity that simply
    # has no events of that kind, so those are filled.
    unmeasurable_cols = [
        "order_velocity_per_hour",
        "refund_velocity_per_hour",
        "login_velocity_per_hour",
        "seller_order_velocity_per_hour",
        "order_burst_ratio",
        "seller_order_burst_ratio",
        "failed_attempt_density",
    ]
    fill_cols = [c for c in temporal_df.columns if c not in unmeasurable_cols]
    temporal_df[fill_cols] = temporal_df[fill_cols].fillna(0)

    # (normalised column, source column, weight)
    signal_spec = [
        ("norm_order_velocity", "order_velocity_per_hour", 0.15),
        ("norm_order_burst", "order_burst_ratio", 0.15),
        ("norm_refund_velocity", "refund_velocity_per_hour", 0.15),
        ("norm_login_velocity", "login_velocity_per_hour", 0.15),
        ("norm_failed_density", "failed_attempt_density", 0.10),
        ("norm_unique_devices", "unique_devices", 0.10),
        ("norm_unique_regions", "unique_regions", 0.05),
        ("norm_seller_velocity", "seller_order_velocity_per_hour", 0.075),
        ("norm_seller_burst", "seller_order_burst_ratio", 0.075),
    ]

    for _, source_col, _ in signal_spec:
        if source_col not in temporal_df.columns:
            temporal_df[source_col] = np.nan if source_col in unmeasurable_cols else 0

    # Report measurability explicitly so an analyst can tell "no burst" from
    # "not enough history to say", and so the suppression is auditable.
    for col in ["order_velocity_per_hour", "refund_velocity_per_hour",
                "login_velocity_per_hour", "seller_order_velocity_per_hour"]:
        measurable = pd.to_numeric(temporal_df[col], errors="coerce").notna()
        temporal_df[col.replace("_per_hour", "_measurable")] = measurable
        log(f"  {col}: {int((~measurable).sum())}/{len(temporal_df)} entities below "
            f"the {MIN_EVENTS_FOR_VELOCITY}-event / {MIN_OBSERVATION_HOURS}h floor -> no velocity signal")

    temporal_score = pd.Series(0.0, index=temporal_df.index)
    for norm_col, source_col, weight in signal_spec:
        values = pd.to_numeric(temporal_df[source_col], errors="coerce")
        # normalize() ranks only the entities where the signal exists; NaN stays NaN.
        temporal_df[norm_col] = normalize(values)
        # An unmeasurable signal contributes nothing while keeping its weight in
        # the denominator. Renormalising per row would do the opposite of the
        # intent: an entity with one event would be scored on its one available
        # signal alone and float back to the top, which is the bug being fixed.
        temporal_score = temporal_score + weight * temporal_df[norm_col].fillna(0.0)

    temporal_df["temporal_risk_score"] = temporal_score

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