"""
Generates sample_data/sales.csv, customers.csv, products.csv for the
AI Data Analyst demo.

Intentionally includes:
  - A handful of statistical outliers in `sales.csv` (revenue / quantity)
    so the anomaly-detection feature has something real to find.
  - Missing values, a few duplicate rows, and one bad date string in
    `sales.csv` so the data-quality checks have something real to flag.
  - A clean month-over-month trend with seasonality so forecasting has
    a signal to project.

Re-run with: python scripts/generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- products
products = pd.DataFrame({
    "product_id": [f"P{100+i}" for i in range(12)],
    "product_name": [
        "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "27in Monitor",
        "Laptop Stand", "Webcam HD", "Noise Cancelling Headphones",
        "Bluetooth Speaker", "Ergonomic Chair", "Standing Desk",
        "Portable SSD 1TB", "Smart Power Strip",
    ],
    "category": [
        "Accessories", "Accessories", "Accessories", "Displays",
        "Accessories", "Accessories", "Audio", "Audio",
        "Furniture", "Furniture", "Storage", "Accessories",
    ],
    "unit_cost": [8.5, 22.0, 14.0, 95.0, 11.0, 18.0, 45.0, 28.0, 120.0, 210.0, 55.0, 12.0],
})

# ----------------------------------------------------------------customers
segments = ["Enterprise", "SMB", "Individual"]
n_customers = 60
customers = pd.DataFrame({
    "customer_id": [f"C{1000+i}" for i in range(n_customers)],
    "customer_name": [f"Customer {i+1}" for i in range(n_customers)],
    "segment": rng.choice(segments, size=n_customers, p=[0.2, 0.35, 0.45]),
    "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=n_customers), unit="D"
    ),
    "region": rng.choice(["North", "South", "East", "West"], size=n_customers),
})

# --------------------------------------------------------------------sales
regions = ["North", "South", "East", "West"]
months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

rows = []
order_id = 1
# base monthly demand per region with a gentle upward trend + seasonality
for region in regions:
    base = {"North": 140, "South": 95, "East": 110, "West": 80}[region]
    for m_idx, month in enumerate(months):
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (month.month / 12))
        trend = 1.0 + 0.01 * m_idx
        n_orders = int(base * seasonal * trend / 6)  # orders that month
        for _ in range(n_orders):
            prod = products.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
            cust = customers[customers["region"] == region].sample(
                1, random_state=rng.integers(0, 1_000_000)
            ).iloc[0]
            qty = int(rng.integers(1, 6))
            unit_price = round(prod["unit_cost"] * rng.uniform(1.4, 2.2), 2)
            day = int(rng.integers(1, 28))
            order_date = month + pd.Timedelta(days=day)
            rows.append({
                "order_id": f"O{order_id:05d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "region": region,
                "customer_id": cust["customer_id"],
                "product_id": prod["product_id"],
                "quantity": qty,
                "unit_price": unit_price,
                "revenue": round(qty * unit_price, 2),
            })
            order_id += 1

sales = pd.DataFrame(rows)

# ---- inject statistical anomalies (unusually large orders) -------------
anomaly_idx = rng.choice(sales.index, size=6, replace=False)
for i in anomaly_idx:
    sales.loc[i, "quantity"] = int(rng.integers(80, 150))
    sales.loc[i, "revenue"] = round(sales.loc[i, "quantity"] * sales.loc[i, "unit_price"], 2)

# ---- inject data quality issues -----------------------------------------
# missing values
missing_idx = rng.choice(sales.index, size=25, replace=False)
sales.loc[missing_idx, "unit_price"] = np.nan

missing_region_idx = rng.choice(sales.index, size=10, replace=False)
sales.loc[missing_region_idx, "region"] = None

# one malformed date
bad_date_idx = rng.choice(sales.index, size=1, replace=False)
sales.loc[bad_date_idx, "order_date"] = "13/45/2024"  # invalid

# a few duplicate rows
dup_rows = sales.sample(4, random_state=7)
sales = pd.concat([sales, dup_rows], ignore_index=True)

# shuffle
sales = sales.sample(frac=1.0, random_state=1).reset_index(drop=True)

products.to_csv(OUT_DIR / "products.csv", index=False)
customers.to_csv(OUT_DIR / "customers.csv", index=False)
sales.to_csv(OUT_DIR / "sales.csv", index=False)

print(f"sales.csv      -> {len(sales)} rows")
print(f"customers.csv  -> {len(customers)} rows")
print(f"products.csv   -> {len(products)} rows")
