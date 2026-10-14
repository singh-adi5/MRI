"""
risk_scoring.py
----------------

This module calculates basic risk scores for customers and sellers in
the online marketplace database. It uses simple rule-based checks
inspired by common fraud patterns to assign risk points and generate
a report of entities that may warrant further investigation.

Rules implemented:

1. **High refund rate (sellers)**
   - A seller with more than 30% of their orders refunded is assigned
     2 risk points. A high refund rate may indicate poor product
     quality or abuse of the refund system.

2. **Order spike (sellers)**
   - For each seller, the script computes the average number of orders
     per day. If on any single day the seller receives more than
     three times this average, the seller is assigned 2 risk points.
     Sudden order spikes can be a signal of artificial demand or
     coordinated manipulation.

3. **Rating manipulation (sellers)**
   - If any customer submits more than three ratings to the same
     seller within a 30 day window, the seller is assigned 2 risk
     points. Repeated ratings from the same customer could indicate
     attempts to skew seller reputation.

4. **Device sharing (customers and sellers)**
   - The login activity table records the device ID used by each
     login. If a device ID is associated with more than three
     distinct user accounts (customers or sellers), each account
     using that device is assigned 3 risk points. Multiple accounts
     on a single device may indicate account farming or collusion.

5. **Multiple failed logins (customers and sellers)**
   - Any user account (customer or seller) with a login record
     indicating more than 3 failed attempts is assigned 1 risk point.
     Numerous failed logins can hint at account takeover attempts or
     brute‑force password guessing.

The script summarises these risk contributions for each entity and
writes the results to ``flagged_entities.csv``. Entities with a
positive risk score are included in the output.

Usage:
    python risk_scoring.py

Ensure you have created the database with ``create_database.py``
before running this script.
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd


def load_tables(db_path: str) -> dict:
    """Load relevant tables from the SQLite database into Pandas DataFrames."""
    conn = sqlite3.connect(db_path)
    try:
        customers = pd.read_sql_query("SELECT * FROM customers", conn)
        sellers = pd.read_sql_query("SELECT * FROM sellers", conn)
        orders = pd.read_sql_query("SELECT * FROM orders", conn)
        refunds = pd.read_sql_query("SELECT * FROM refunds", conn)
        ratings = pd.read_sql_query("SELECT * FROM ratings", conn)
        login = pd.read_sql_query("SELECT * FROM login_activity", conn)
    finally:
        conn.close()
    return {
        "customers": customers,
        "sellers": sellers,
        "orders": orders,
        "refunds": refunds,
        "ratings": ratings,
        "login": login,
    }


def compute_refund_rate_scores(orders: pd.DataFrame, refunds: pd.DataFrame) -> dict:
    """Compute risk points for sellers with a high refund rate."""
    # Count orders and refunds per seller
    orders_per_seller = orders.groupby("seller_id").size()
    refunds_merged = refunds.merge(orders[["order_id", "seller_id"]], on="order_id", how="left")
    refunds_per_seller = refunds_merged.groupby("seller_id").size()

    risk_scores = {}
    for seller_id, total_orders in orders_per_seller.items():
        num_refunds = refunds_per_seller.get(seller_id, 0)
        refund_rate = num_refunds / total_orders if total_orders > 0 else 0
        if refund_rate > 0.3:
            risk_scores[seller_id] = {
                "score": 2,
                "reasons": [
                    f"High refund rate ({num_refunds}/{total_orders} = {refund_rate:.2%})"
                ],
            }
    return risk_scores


def compute_order_spike_scores(orders: pd.DataFrame) -> dict:
    """Compute risk points for sellers experiencing order spikes."""
    risk_scores = {}
    # Convert order_date to date only (YYYY-MM-DD)
    orders["order_day"] = pd.to_datetime(orders["order_date"]).dt.date
    # Determine average orders per day for each seller
    daily_counts = orders.groupby(["seller_id", "order_day"]).size().reset_index(name="count")
    avg_per_day = daily_counts.groupby("seller_id")["count"].mean()
    # For each seller, check if any day exceeds 3 * average
    for seller_id, avg_count in avg_per_day.items():
        # Skip sellers with low average (avoid division by zero or spurious spikes)
        if avg_count == 0:
            continue
        # Get counts for seller
        seller_counts = daily_counts[daily_counts["seller_id"] == seller_id]
        if any(seller_counts["count"] > 3 * avg_count):
            max_count = seller_counts["count"].max()
            risk_scores[seller_id] = {
                "score": 2,
                "reasons": [
                    f"Order spike detected (max {max_count} orders vs average {avg_count:.2f} per day)"
                ],
            }
    return risk_scores


def compute_rating_manipulation_scores(ratings: pd.DataFrame) -> dict:
    """Compute risk points for sellers with potential rating manipulation."""
    # Convert rating_date to datetime
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"])
    # Sort by date for windowing
    ratings = ratings.sort_values(by="rating_date")
    risk_scores = {}
    # Group by seller and customer and use rolling window
    grouped = ratings.groupby(["seller_id", "customer_id"])
    for (seller_id, customer_id), group in grouped:
        # compute counts within 30-day rolling window
        group = group.sort_values("rating_date")
        for i in range(len(group)):
            start_date = group.iloc[i]["rating_date"]
            end_date = start_date + timedelta(days=30)
            window_count = ((group["rating_date"] >= start_date) & (group["rating_date"] <= end_date)).sum()
            if window_count > 3:
                # mark seller once
                risk_scores.setdefault(seller_id, {"score": 0, "reasons": []})
                risk_scores[seller_id]["score"] += 2
                risk_scores[seller_id]["reasons"].append(
                    f"Rating manipulation (customer {customer_id} submitted {window_count} ratings within 30 days)"
                )
                break  # break after first detection for this pair
    return risk_scores


def compute_device_sharing_scores(login: pd.DataFrame) -> dict:
    """Compute risk points for users logging in from shared devices."""
    # Count unique users per device
    user_counts_by_device = login.groupby("device_id")["user_id"].nunique()
    risk_scores = {}
    # Identify devices used by more than 3 users
    flagged_devices = user_counts_by_device[user_counts_by_device > 3].index
    for device_id in flagged_devices:
        users_on_device = login[login["device_id"] == device_id]["user_id"].unique()
        for user_id in users_on_device:
            risk_scores.setdefault(user_id, {"score": 0, "reasons": []})
            risk_scores[user_id]["score"] += 3
            risk_scores[user_id]["reasons"].append(
                f"Device sharing: device {device_id} used by {user_counts_by_device[device_id]} accounts"
            )
    return risk_scores


def compute_failed_login_scores(login: pd.DataFrame) -> dict:
    """Compute risk points for users with multiple failed logins."""
    risk_scores = {}
    # Identify users with >3 failed attempts in any login record
    high_failed = login[login["failed_attempts"] > 3]["user_id"].unique()
    for user_id in high_failed:
        risk_scores[user_id] = {
            "score": 1,
            "reasons": ["Multiple failed login attempts (>3)"],
        }
    return risk_scores


def aggregate_scores(*score_dicts) -> dict:
    """Combine multiple risk score dictionaries into a single dictionary.

    Each dictionary should map an entity ID to a dict with ``score`` and
    ``reasons`` keys. Scores and reasons will be summed and extended
    respectively for overlapping entities.
    """
    combined = defaultdict(lambda: {"score": 0, "reasons": []})
    for score_dict in score_dicts:
        for entity_id, info in score_dict.items():
            combined[entity_id]["score"] += info["score"]
            combined[entity_id]["reasons"].extend(info["reasons"])
    return combined


def write_flagged_entities(risk_scores: dict, output_path: str):
    """Write the risk scores to a CSV file, including entity type and reasons."""
    rows = []
    for entity_id, info in risk_scores.items():
        if info["score"] <= 0:
            continue
        # Determine type from prefix (C for customer, S for seller) if present
        # Otherwise treat numeric IDs as sellers (orders etc.)
        if isinstance(entity_id, str):
            if entity_id.startswith("C"):
                entity_type = "customer"
                numeric_id = entity_id[1:]
            elif entity_id.startswith("S"):
                entity_type = "seller"
                numeric_id = entity_id[1:]
            else:
                entity_type = "unknown"
                numeric_id = entity_id
        else:
            # numeric ID for seller
            entity_type = "seller"
            numeric_id = str(entity_id)
        rows.append(
            {
                "entity_id": numeric_id,
                "entity_type": entity_type,
                "risk_score": info["score"],
                "reasons": "; ".join(info["reasons"]),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(by="risk_score", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Risk report written to: {output_path}")


def main():
    db_path = os.path.join(os.path.dirname(__file__), "marketplace.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database file not found at {db_path}. Please run create_database.py first."
        )

    tables = load_tables(db_path)
    orders = tables["orders"].copy()
    refunds = tables["refunds"].copy()
    ratings = tables["ratings"].copy()
    login = tables["login"].copy()

    # Compute individual risk scores
    refund_scores = compute_refund_rate_scores(orders, refunds)
    spike_scores = compute_order_spike_scores(orders)
    rating_scores = compute_rating_manipulation_scores(ratings)
    device_scores = compute_device_sharing_scores(login)
    failed_login_scores = compute_failed_login_scores(login)

    # Combine scores
    combined = aggregate_scores(refund_scores, spike_scores, rating_scores, device_scores, failed_login_scores)

    # Write output
    output_path = os.path.join(os.path.dirname(__file__), "flagged_entities.csv")
    write_flagged_entities(combined, output_path)


if __name__ == "__main__":
    main()