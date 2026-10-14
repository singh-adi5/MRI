"""
generate_data.py
-----------------

This script generates sample data for an online marketplace project.
It produces CSV files for customers, sellers, orders, refunds,
ratings and login activity. These files can be loaded into a SQLite
database for analysis, or analysed directly using Pandas or any
other data analysis tool.

The generated data is synthetic and intended purely for learning
purposes. Each CSV file will be written into the ``data`` directory
relative to this script's location.
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd

from mri.config import RAW_DATA_DIR


def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between ``start`` and ``end``."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_customers(num_customers: int) -> pd.DataFrame:
    """Generate a customers DataFrame with synthetic data."""
    regions = ["Europe", "North America", "South America", "Asia", "Africa", "Oceania"]
    device_types = ["mobile", "desktop", "tablet"]
    today = datetime(2026, 3, 15)
    one_year_ago = today - timedelta(days=365)

    data = []
    for customer_id in range(1, num_customers + 1):
        signup_dt = random_date(one_year_ago, today)
        region = random.choice(regions)
        device = random.choice(device_types)
        account_age_days = (today - signup_dt).days
        data.append(
            {
                "customer_id": customer_id,
                "signup_date": signup_dt.strftime("%Y-%m-%d"),
                "region": region,
                "device_type": device,
                "account_age_days": account_age_days,
            }
        )
    return pd.DataFrame(data)


def generate_sellers(num_sellers: int) -> pd.DataFrame:
    """Generate a sellers DataFrame with synthetic data."""
    categories = ["electronics", "fashion", "home", "beauty", "sports", "toys", "books"]
    regions = ["Europe", "North America", "South America", "Asia", "Africa", "Oceania"]
    today = datetime(2026, 3, 15)
    two_years_ago = today - timedelta(days=730)
    data = []
    for seller_id in range(1, num_sellers + 1):
        join_dt = random_date(two_years_ago, today)
        region = random.choice(regions)
        category = random.choice(categories)
        # Randomly mark some sellers as having a past risk flag
        risk_flag_history = random.choice(["none", "minor", "major"])
        data.append(
            {
                "seller_id": seller_id,
                "join_date": join_dt.strftime("%Y-%m-%d"),
                "seller_region": region,
                "category": category,
                "risk_flag_history": risk_flag_history,
            }
        )
    return pd.DataFrame(data)


def generate_orders(num_orders: int, num_customers: int, num_sellers: int, num_products: int) -> pd.DataFrame:
    """Generate an orders DataFrame with synthetic data."""
    today = datetime(2026, 3, 15)
    three_months_ago = today - timedelta(days=90)
    payment_statuses = ["completed", "pending", "cancelled"]
    data = []
    for order_id in range(1, num_orders + 1):
        order_dt = random_date(three_months_ago, today)
        customer_id = random.randint(1, num_customers)
        seller_id = random.randint(1, num_sellers)
        product_id = random.randint(1, num_products)
        amount = round(random.uniform(5, 500), 2)
        payment_status = random.choices(
            payment_statuses, weights=[0.85, 0.1, 0.05], k=1
        )[0]
        data.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "seller_id": seller_id,
                "product_id": product_id,
                "order_date": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "payment_status": payment_status,
            }
        )
    return pd.DataFrame(data)


def generate_refunds(orders_df: pd.DataFrame, refund_probability: float = 0.2) -> pd.DataFrame:
    """Generate a refunds DataFrame based on a subset of orders.

    refund_probability controls the proportion of orders that receive a refund.
    """
    refund_reasons = ["defective", "late delivery", "changed mind", "fraud"]
    refund_statuses = ["approved", "pending", "rejected"]
    data = []
    refund_id = 1
    for _, row in orders_df.iterrows():
        if random.random() < refund_probability:
            refund_dt = datetime.strptime(row["order_date"], "%Y-%m-%d %H:%M:%S") + timedelta(
                days=random.randint(1, 30)
            )
            refund_reason = random.choice(refund_reasons)
            refund_status = random.choices(
                refund_statuses, weights=[0.7, 0.2, 0.1], k=1
            )[0]
            data.append(
                {
                    "refund_id": refund_id,
                    "order_id": row["order_id"],
                    "refund_date": refund_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "refund_reason": refund_reason,
                    "refund_status": refund_status,
                }
            )
            refund_id += 1
    return pd.DataFrame(data)


def generate_ratings(num_ratings: int, num_customers: int, num_sellers: int) -> pd.DataFrame:
    """Generate a ratings DataFrame with synthetic data."""
    today = datetime(2026, 3, 15)
    three_months_ago = today - timedelta(days=90)
    data = []
    for rating_id in range(1, num_ratings + 1):
        customer_id = random.randint(1, num_customers)
        seller_id = random.randint(1, num_sellers)
        rating_value = random.randint(1, 5)
        rating_dt = random_date(three_months_ago, today)
        data.append(
            {
                "rating_id": rating_id,
                "customer_id": customer_id,
                "seller_id": seller_id,
                "rating_value": rating_value,
                "rating_date": rating_dt.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return pd.DataFrame(data)


def generate_login_activity(num_records: int, num_customers: int, num_sellers: int) -> pd.DataFrame:
    """Generate a login activity DataFrame with synthetic data for both customers and sellers."""
    regions = ["Europe", "North America", "South America", "Asia", "Africa", "Oceania"]
    devices = ["deviceA", "deviceB", "deviceC", "deviceD", "deviceE"]
    today = datetime(2026, 3, 15)
    three_months_ago = today - timedelta(days=90)
    data = []
    for _ in range(num_records):
        # half of the login records will be customers, half sellers
        if random.random() < 0.5:
            user_id = f"C{random.randint(1, num_customers)}"
        else:
            user_id = f"S{random.randint(1, num_sellers)}"
        login_dt = random_date(three_months_ago, today)
        ip_region = random.choice(regions)
        device_id = random.choice(devices)
        failed_attempts = random.choices([0, 1, 2, 3, 4, 5], weights=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05], k=1)[0]
        data.append(
            {
                "user_id": user_id,
                "login_date": login_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "ip_region": ip_region,
                "device_id": device_id,
                "failed_attempts": failed_attempts,
            }
        )
    return pd.DataFrame(data)


def main():
    # Configuration for dataset sizes
    NUM_CUSTOMERS = 300
    NUM_SELLERS = 50
    NUM_PRODUCTS = 200
    NUM_ORDERS = 1000
    NUM_RATINGS = 800
    NUM_LOGIN_RECORDS = 1000

    base_dir = str(RAW_DATA_DIR)
    os.makedirs(base_dir, exist_ok=True)

    # Generate each dataset
    customers_df = generate_customers(NUM_CUSTOMERS)
    sellers_df = generate_sellers(NUM_SELLERS)
    orders_df = generate_orders(NUM_ORDERS, NUM_CUSTOMERS, NUM_SELLERS, NUM_PRODUCTS)
    refunds_df = generate_refunds(orders_df)
    ratings_df = generate_ratings(NUM_RATINGS, NUM_CUSTOMERS, NUM_SELLERS)
    login_df = generate_login_activity(NUM_LOGIN_RECORDS, NUM_CUSTOMERS, NUM_SELLERS)

    # Write to CSV files
    customers_df.to_csv(os.path.join(base_dir, "customers.csv"), index=False)
    sellers_df.to_csv(os.path.join(base_dir, "sellers.csv"), index=False)
    orders_df.to_csv(os.path.join(base_dir, "orders.csv"), index=False)
    refunds_df.to_csv(os.path.join(base_dir, "refunds.csv"), index=False)
    ratings_df.to_csv(os.path.join(base_dir, "ratings.csv"), index=False)
    login_df.to_csv(os.path.join(base_dir, "login_activity.csv"), index=False)

    print(f"Data generated and saved to: {base_dir}")


if __name__ == "__main__":
    main()