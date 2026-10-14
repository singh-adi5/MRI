import os
import sqlite3
import pandas as pd
import numpy as np

from mri.config import ROOT as BASE_DIR  # repository root
DB_PATH = os.path.join(BASE_DIR, "data", "marketplace.db")
FLAGGED_CSV = os.path.join(BASE_DIR, "outputs", "flagged_entities.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUT_PATH = os.path.join(ARTIFACT_DIR, "entity_features.csv")


def log(msg: str):
    print(f"[feature_eng] {msg}")


def load_table(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def ensure_columns(df: pd.DataFrame, required_cols: list, default_value=0) -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = default_value
    return df


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.fillna(0)


def normalize_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
    return df


def build_customer_account_age(customers: pd.DataFrame) -> pd.DataFrame:
    if "account_age_days" in customers.columns:
        customers["account_age_days"] = pd.to_numeric(
            customers["account_age_days"], errors="coerce"
        ).fillna(0)
        return customers

    if "signup_date" in customers.columns:
        customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
        customers["account_age_days"] = (
            pd.Timestamp.now().normalize() - customers["signup_date"]
        ).dt.days.fillna(0)
        customers["account_age_days"] = customers["account_age_days"].clip(lower=0)
        return customers

    customers["account_age_days"] = 0
    return customers


def load_flagged_entities() -> pd.DataFrame:
    if os.path.exists(FLAGGED_CSV):
        flagged = pd.read_csv(FLAGGED_CSV)
        flagged["entity_id"] = flagged["entity_id"].astype(str)
        flagged["entity_type"] = flagged["entity_type"].astype(str)
        return flagged
    return pd.DataFrame(columns=["entity_id", "entity_type"])


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log("Loading source tables from SQLite...")
    customers = load_table("SELECT * FROM customers")
    sellers = load_table("SELECT * FROM sellers")
    orders = load_table("SELECT * FROM orders")
    refunds = load_table("SELECT * FROM refunds")
    ratings = load_table("SELECT * FROM ratings")
    login = load_table("SELECT * FROM login_activity")
    flagged = load_flagged_entities()

    # Normalize dates
    orders = normalize_datetime(orders, "order_date")
    refunds = normalize_datetime(refunds, "refund_date")
    ratings = normalize_datetime(ratings, "rating_date")
    login = normalize_datetime(login, "login_date")
    customers = normalize_datetime(customers, "signup_date")
    sellers = normalize_datetime(sellers, "join_date")

    # Defensive numeric coercion
    for df, cols in [
        (orders, ["amount"]),
        (ratings, ["rating_value"]),
        (login, ["failed_attempts"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    customers = build_customer_account_age(customers)

    # -------------------------
    # Seller-level features
    # -------------------------
    log("Building seller-level features...")

    seller_orders = (
        orders.groupby("seller_id", dropna=False)
        .agg(
            total_orders=("order_id", "count"),
            avg_order_value=("amount", "mean"),
            total_revenue=("amount", "sum"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    refund_seller_join = refunds.merge(
        orders[["order_id", "seller_id"]],
        on="order_id",
        how="left"
    )

    seller_refunds = (
        refund_seller_join.groupby("seller_id", dropna=False)
        .agg(total_refunds=("refund_id", "count"))
        .reset_index()
    )

    seller_ratings = (
        ratings.groupby("seller_id", dropna=False)
        .agg(
            avg_rating=("rating_value", "mean"),
            rating_count=("rating_id", "count"),
            rating_std=("rating_value", "std"),
        )
        .reset_index()
    )

    if "order_date" in orders.columns:
        orders["order_day"] = orders["order_date"].dt.date
        seller_daily = (
            orders.groupby(["seller_id", "order_day"], dropna=False)
            .size()
            .reset_index(name="daily_orders")
        )
        seller_spikes = (
            seller_daily.groupby("seller_id", dropna=False)
            .agg(
                max_daily_orders=("daily_orders", "max"),
                avg_daily_orders=("daily_orders", "mean"),
            )
            .reset_index()
        )
        seller_spikes["order_spike_ratio"] = safe_divide(
            seller_spikes["max_daily_orders"],
            seller_spikes["avg_daily_orders"]
        )
    else:
        seller_spikes = pd.DataFrame(columns=[
            "seller_id", "max_daily_orders", "avg_daily_orders", "order_spike_ratio"
        ])

    seller_features = (
        sellers.merge(seller_orders, on="seller_id", how="left")
               .merge(seller_refunds, on="seller_id", how="left")
               .merge(seller_ratings, on="seller_id", how="left")
               .merge(seller_spikes, on="seller_id", how="left")
    )

    seller_features["total_refunds"] = seller_features["total_refunds"].fillna(0)
    seller_features["refund_rate"] = safe_divide(
        seller_features["total_refunds"],
        seller_features["total_orders"].fillna(0)
    )

    seller_features["entity_id"] = seller_features["seller_id"].astype(str)
    seller_features["entity_type"] = "seller"

    # -------------------------
    # Customer-level features
    # -------------------------
    log("Building customer-level features...")

    customer_orders = (
        orders.groupby("customer_id", dropna=False)
        .agg(
            total_orders=("order_id", "count"),
            avg_order_value=("amount", "mean"),
            total_spend=("amount", "sum"),
            unique_sellers=("seller_id", "nunique"),
        )
        .reset_index()
    )

    refund_customer_join = refunds.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="left"
    )

    customer_refunds = (
        refund_customer_join.groupby("customer_id", dropna=False)
        .agg(total_refunds=("refund_id", "count"))
        .reset_index()
    )

    customer_ratings = (
        ratings.groupby("customer_id", dropna=False)
        .agg(
            total_ratings=("rating_id", "count"),
            avg_rating_given=("rating_value", "mean"),
        )
        .reset_index()
    )

    login["user_id"] = login["user_id"].astype(str)
    login["entity_type"] = login["user_id"].str[0].map({"C": "customer", "S": "seller"})
    login["numeric_id"] = login["user_id"].str[1:]

    customer_login = (
        login[login["entity_type"] == "customer"]
        .groupby("numeric_id", dropna=False)
        .agg(
            login_events=("user_id", "count"),
            avg_failed_attempts=("failed_attempts", "mean"),
            max_failed_attempts=("failed_attempts", "max"),
            unique_devices=("device_id", "nunique"),
            unique_regions=("ip_region", "nunique"),
        )
        .reset_index()
        .rename(columns={"numeric_id": "customer_id"})
    )

    if not customer_login.empty:
        customer_login["customer_id"] = pd.to_numeric(
            customer_login["customer_id"], errors="coerce"
        )
        customer_login = customer_login.dropna(subset=["customer_id"])
        customer_login["customer_id"] = customer_login["customer_id"].astype(int)

    customer_features = (
        customers.merge(customer_orders, on="customer_id", how="left")
                 .merge(customer_refunds, on="customer_id", how="left")
                 .merge(customer_ratings, on="customer_id", how="left")
                 .merge(customer_login, on="customer_id", how="left")
    )

    customer_features["total_refunds"] = customer_features["total_refunds"].fillna(0)
    customer_features["refund_rate"] = safe_divide(
        customer_features["total_refunds"],
        customer_features["total_orders"].fillna(0)
    )

    customer_features["entity_id"] = customer_features["customer_id"].astype(str)
    customer_features["entity_type"] = "customer"

    # -------------------------
    # Required columns
    # -------------------------
    seller_required = [
        "entity_id", "entity_type",
        "total_orders", "avg_order_value", "total_revenue", "unique_customers",
        "total_refunds", "refund_rate",
        "avg_rating", "rating_count", "rating_std",
        "max_daily_orders", "avg_daily_orders", "order_spike_ratio"
    ]

    customer_required = [
        "entity_id", "entity_type",
        "total_orders", "avg_order_value", "total_spend", "unique_sellers",
        "total_refunds", "refund_rate",
        "total_ratings", "avg_rating_given",
        "login_events", "avg_failed_attempts", "max_failed_attempts",
        "unique_devices", "unique_regions", "account_age_days"
    ]

    seller_features = ensure_columns(seller_features, seller_required, default_value=0)
    customer_features = ensure_columns(customer_features, customer_required, default_value=0)

    seller_export = seller_features[seller_required].copy()
    customer_export = customer_features[customer_required].copy()

    # unify feature space
    all_features = pd.concat([seller_export, customer_export], ignore_index=True, sort=False)
    all_features = all_features.fillna(0)

    # -------------------------
    # Weak labels from V1
    # -------------------------
    log("Applying weak labels from V1 flagged entities...")
    all_features["label"] = 0

    if not flagged.empty and {"entity_id", "entity_type"}.issubset(flagged.columns):
        flagged_keys = set(zip(flagged["entity_id"], flagged["entity_type"]))
        all_features["label"] = all_features.apply(
            lambda row: 1 if (str(row["entity_id"]), str(row["entity_type"])) in flagged_keys else 0,
            axis=1
        )

    # -------------------------
    # Final cleanup
    # -------------------------
    numeric_cols = [c for c in all_features.columns if c not in ["entity_id", "entity_type"]]
    for col in numeric_cols:
        all_features[col] = pd.to_numeric(all_features[col], errors="coerce").fillna(0)

    all_features = all_features.drop_duplicates(subset=["entity_id", "entity_type"]).reset_index(drop=True)

    all_features.to_csv(OUTPUT_PATH, index=False)

    log(f"Feature dataset saved to: {OUTPUT_PATH}")
    log(f"Shape: {all_features.shape}")
    log("Sample:")
    print(all_features.head())


if __name__ == "__main__":
    main()