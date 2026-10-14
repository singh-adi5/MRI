# Data Dictionary for the Online Marketplace Project

This document describes the columns contained in each of the synthetic
CSV files used in the fraud pattern analysis project. Understanding
the schema will help you write your own queries or extend the
analysis.

## customers.csv

| Column            | Type    | Description                                        |
|------------------ |-------- |----------------------------------------------------|
| **customer_id**    | int     | Unique identifier for the customer.                |
| **signup_date**    | date    | Date the customer signed up for the platform.      |
| **region**         | string  | Geographic region of the customer (e.g., Europe). |
| **device_type**    | string  | Device type used during sign up.                  |
| **account_age_days** | int   | Days since sign up at the current date.            |

## sellers.csv

| Column            | Type    | Description                                              |
|------------------ |-------- |----------------------------------------------------------|
| **seller_id**      | int     | Unique identifier for the seller.                        |
| **join_date**      | date    | Date the seller joined the platform.                     |
| **seller_region**  | string  | Geographic region of the seller.                        |
| **category**       | string  | Category of products sold (e.g., electronics, fashion). |
| **risk_flag_history** | string | Prior risk classification (none, minor, major).        |

## orders.csv

| Column         | Type    | Description                                                  |
|--------------- |-------- |--------------------------------------------------------------|
| **order_id**    | int     | Unique order identifier.                                     |
| **customer_id** | int     | ID of the customer who placed the order.                    |
| **seller_id**   | int     | ID of the seller who received the order.                    |
| **product_id**  | int     | Identifier of the product.                                   |
| **order_date**  | datetime | Timestamp of the order (YYYY‑MM‑DD HH:MM:SS).               |
| **amount**      | float   | Monetary value of the order.                                |
| **payment_status** | string | Status of payment (completed, pending, cancelled).          |

## refunds.csv

| Column        | Type    | Description                                                 |
|-------------- |-------- |-------------------------------------------------------------|
| **refund_id**  | int     | Unique refund identifier.                                   |
| **order_id**   | int     | ID of the order associated with the refund.                |
| **refund_date** | datetime | Timestamp of the refund request.                           |
| **refund_reason** | string | Reason for the refund (defective, late delivery, etc.).    |
| **refund_status** | string | Status of the refund request (approved, pending, rejected). |

## ratings.csv

| Column         | Type    | Description                                                   |
|--------------- |-------- |---------------------------------------------------------------|
| **rating_id**   | int     | Unique rating identifier.                                    |
| **customer_id** | int     | ID of the customer leaving the rating.                       |
| **seller_id**   | int     | ID of the seller being rated.                                |
| **rating_value** | int    | Rating value (1–5 stars).                                    |
| **rating_date** | datetime | Timestamp of the rating.                                     |

## login_activity.csv

| Column           | Type    | Description                                                                |
|----------------- |-------- |----------------------------------------------------------------------------|
| **user_id**       | string  | Identifier of the logging user. Customers are prefixed with `C`, sellers with `S`. |
| **login_date**    | datetime | Timestamp of the login event.                                              |
| **ip_region**     | string  | Geographic region of the IP address used for login.                        |
| **device_id**     | string  | Identifier of the device used.                                            |
| **failed_attempts** | int    | Number of failed login attempts immediately preceding this login event.     |

---

These datasets form the basis for loading into the SQLite database and
performing risk calculations as described in the project overview. Feel
free to extend the schema or add additional attributes to suit your
own analysis requirements.