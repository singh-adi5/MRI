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
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

from mri.config import DB_PATH, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Canonical entity keys
# ---------------------------------------------------------------------------
# Scorers draw IDs from two different shapes of source data:
#   * orders/refunds/ratings carry a numeric seller_id (e.g. 17)
#   * login_activity carries a prefixed user_id (e.g. "S17", "C225")
# Mixing those in one dict silently splits an entity in two, because
# 17 != "S17" as dict keys. Every scorer therefore emits the SAME canonical
# key -- "<TYPE-PREFIX><numeric id>", e.g. "S17" -- and aggregation rejects
# anything else, so a future scorer cannot reintroduce the collision quietly.

CUSTOMER_PREFIX = "C"
SELLER_PREFIX = "S"
UNKNOWN_PREFIX = "X"

_CANONICAL_RE = re.compile(r"^[CS][0-9]+$")
_PREFIXED_RE = re.compile(r"^([A-Za-z])\s*([0-9]+)$")


def entity_key(raw, entity_type: str = None) -> str:
    """Return the canonical key for an entity ID coming from any source table.

    ``raw`` may be an int (17), a numeric string ("17") or an already prefixed
    string ("S17" / "s17"). ``entity_type`` ("seller"/"customer") is required
    when ``raw`` carries no prefix of its own.

    Unrecognised IDs are preserved under an explicit "X" namespace rather than
    being coerced into a seller/customer bucket -- an unknown entity must stay
    visible, not be silently mislabelled.
    """
    text = str(raw).strip()
    if not text:
        raise ValueError("empty entity id")

    prefix_by_type = {"customer": CUSTOMER_PREFIX, "seller": SELLER_PREFIX}

    if text.isdigit():
        if entity_type not in prefix_by_type:
            raise ValueError(
                f"numeric entity id {text!r} needs entity_type in {sorted(prefix_by_type)}"
            )
        return f"{prefix_by_type[entity_type]}{int(text)}"

    match = _PREFIXED_RE.match(text)
    if match:
        prefix, digits = match.group(1).upper(), match.group(2)
        if prefix in (CUSTOMER_PREFIX, SELLER_PREFIX):
            # int() drops any zero padding so "S017" and 17 land on one key.
            return f"{prefix}{int(digits)}"

    return f"{UNKNOWN_PREFIX}:{text}"


def split_entity_key(key: str):
    """Inverse of :func:`entity_key`: canonical key -> (entity_id, entity_type)."""
    if _CANONICAL_RE.match(key):
        entity_type = "customer" if key[0] == CUSTOMER_PREFIX else "seller"
        return key[1:], entity_type
    if key.startswith(f"{UNKNOWN_PREFIX}:"):
        return key[len(UNKNOWN_PREFIX) + 1:], "unknown"
    raise ValueError(f"not a canonical entity key: {key!r}")


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
            # canonical key: orders.seller_id is numeric, login user_id is "S17"
            risk_scores[entity_key(seller_id, "seller")] = {
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
            risk_scores[entity_key(seller_id, "seller")] = {
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
                key = entity_key(seller_id, "seller")
                risk_scores.setdefault(key, {"score": 0, "reasons": []})
                risk_scores[key]["score"] += 2
                risk_scores[key]["reasons"].append(
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
            # login user_ids are already prefixed ("S17"); normalise anyway so
            # padding/case variants cannot split one entity across two keys.
            key = entity_key(user_id)
            risk_scores.setdefault(key, {"score": 0, "reasons": []})
            risk_scores[key]["score"] += 3
            risk_scores[key]["reasons"].append(
                f"Device sharing: device {device_id} used by {user_counts_by_device[device_id]} accounts"
            )
    return risk_scores


def compute_failed_login_scores(login: pd.DataFrame) -> dict:
    """Compute risk points for users with multiple failed logins."""
    risk_scores = {}
    # Identify users with >3 failed attempts in any login record
    high_failed = login[login["failed_attempts"] > 3]["user_id"].unique()
    for user_id in high_failed:
        risk_scores[entity_key(user_id)] = {
            "score": 1,
            "reasons": ["Multiple failed login attempts (>3)"],
        }
    return risk_scores


def aggregate_scores(*score_dicts) -> dict:
    """Combine multiple risk score dictionaries into a single dictionary.

    Each dictionary must map a CANONICAL entity key (see :func:`entity_key`)
    to a dict with ``score`` and ``reasons`` keys. Scores and reasons are
    summed and extended respectively for overlapping entities.

    The key shape is asserted here rather than coerced: a scorer that emits a
    raw numeric ID would otherwise aggregate into its own separate entity and
    halve both rows' scores, which is exactly the defect this guard prevents.
    """
    combined = defaultdict(lambda: {"score": 0, "reasons": []})
    for score_dict in score_dicts:
        for entity_id, info in score_dict.items():
            if not isinstance(entity_id, str):
                raise TypeError(
                    f"non-canonical entity key {entity_id!r} ({type(entity_id).__name__}); "
                    "scorers must emit entity_key(...) output"
                )
            split_entity_key(entity_id)  # raises on a malformed key
            combined[entity_id]["score"] += info["score"]
            combined[entity_id]["reasons"].extend(info["reasons"])
    return combined


def write_flagged_entities(risk_scores: dict, output_path: str):
    """Write the risk scores to a CSV file, including entity type and reasons."""
    rows = []
    unknown = 0
    for key, info in risk_scores.items():
        if info["score"] <= 0:
            continue
        # Keys are canonical by construction; the writer only unpacks them.
        numeric_id, entity_type = split_entity_key(key)
        if entity_type == "unknown":
            unknown += 1
        rows.append(
            {
                "entity_id": numeric_id,
                "entity_type": entity_type,
                "risk_score": info["score"],
                "reasons": "; ".join(info["reasons"]),
            }
        )
    df = pd.DataFrame(rows, columns=["entity_id", "entity_type", "risk_score", "reasons"])

    # One row per entity is a hard invariant of this report: downstream weak
    # labelling joins on (entity_id, entity_type), so a split entity both
    # under-scores the entity and double-counts it in the label set.
    dupes = df.duplicated(subset=["entity_id", "entity_type"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (entity_id, entity_type) rows in risk report")

    if unknown:
        print(f"WARNING: {unknown} flagged entities could not be typed (entity_type=unknown)")

    df = df.sort_values(by="risk_score", ascending=False)
    df.to_csv(output_path, index=False)
    counts = df["entity_type"].value_counts().to_dict()
    print(f"Risk report written to: {output_path} ({len(df)} rows: {counts})")


def main():
    db_path = str(DB_PATH)
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database file not found at {db_path}. Run `python -m mri.data.build_db` first."
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # outputs/ is generated, may not exist yet
    output_path = os.path.join(OUTPUT_DIR, "flagged_entities.csv")
    write_flagged_entities(combined, output_path)


if __name__ == "__main__":
    main()