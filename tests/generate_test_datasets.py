"""
Deterministic Synthetic Test Dataset Generator for Local-First Autonomous ML Platform.
Covers all 10 synthetic test scenarios defined in Section 15 of ORIGINAL_REQUEST.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl


# ==============================================================================
# Scenario 1: Small Classification Dataset
# ==============================================================================
def generate_small_classification(output_dir: Path, seed: int = 42) -> Path:
    """Generate small_classification.csv: ~1,000 rows, 10 columns, balanced churn."""
    rng = np.random.default_rng(seed)
    n = 1000

    age = rng.integers(18, 76, n)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20.0, 120.0, n), 2)
    total = np.round(tenure * monthly * rng.uniform(0.95, 1.05, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    support_tickets = rng.poisson(1.5, n)

    # Balanced churn target (intercept -0.7 yields ~50/50 balance)
    z = -0.7 + (contract == "Month-to-month") * 1.2 - (tenure / 36.0) * 1.0 + (monthly / 100.0) * 0.8 + support_tickets * 0.3
    prob = 1.0 / (1.0 + np.exp(-z))
    churn = (rng.random(n) < prob).astype(int)

    df = pl.DataFrame({
        "customer_id": [f"CUST_{i+1:04d}" for i in range(n)],
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "support_tickets": support_tickets,
        "churn": churn
    })

    out_path = output_dir / "small_classification.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 2: Large CSV Dataset (>= 50 MB / >= 200,000 rows)
# ==============================================================================
def generate_large_dataset(output_dir: Path, row_count: int = 220_000, seed: int = 42) -> Path:
    """Generate large_dataset.csv: >= 50 MB and >= 200,000 rows for out-of-core testing."""
    rng = np.random.default_rng(seed)
    n = max(row_count, 220_000)

    # Generate columns with descriptive text to reach >= 50 MB on disk
    tiers = [
        "enterprise_diamond_tier_account",
        "business_gold_tier_account",
        "retail_standard_account",
        "freemium_tier_starter_account"
    ]
    categories = [
        "grocery_supermarket_retail",
        "online_digital_entertainment",
        "travel_airline_hospitality",
        "petrol_convenience_fuel",
        "healthcare_pharmacy_wellness"
    ]
    regions = [
        "north_america_east_region_dc01",
        "north_america_west_region_dc02",
        "europe_central_region_dc03",
        "asia_pacific_region_dc04",
        "latin_america_region_dc05"
    ]
    memos = [
        "Automated enterprise recurring settlement clearance batch verification process reference code",
        "Customer payment authorization via point-of-sale terminal network gateway transaction log",
        "Cross-border digital settlement remittance clearance and currency exchange conversion ledger",
        "Merchant settlement escrow deposit credit adjustment and accounting reconciliation notice"
    ]

    amount = np.round(rng.uniform(1.0, 5000.0, n), 2)
    fee = np.round(amount * 0.025 + 0.30, 2)
    balance_before = np.round(rng.uniform(10.0, 100000.0, n), 2)
    balance_after = np.round(balance_before - amount - fee, 2)
    risk_score = np.round(rng.uniform(0.0, 1.0, n), 4)
    latency = np.round(rng.uniform(5.0, 800.0, n), 1)
    freq_1h = rng.poisson(3, n)
    freq_24h = rng.poisson(15, n)
    is_foreign = rng.choice([0, 1], n, p=[0.85, 0.15])
    auth_factors = rng.choice([1, 2, 3], n, p=[0.70, 0.25, 0.05])

    # Latent fraud target (~8% positive)
    z = -2.5 + (amount > 2000.0) * 1.5 + risk_score * 2.0 + is_foreign * 1.0
    prob = 1.0 / (1.0 + np.exp(-z))
    target = (rng.random(n) < prob).astype(int)

    df = pl.DataFrame({
        "transaction_id": [f"TXN_{i+1:07d}" for i in range(n)],
        "timestamp": ["2023-06-15T12:00:00Z"] * n,
        "customer_tier": rng.choice(tiers, n),
        "merchant_category": rng.choice(categories, n),
        "geo_region": rng.choice(regions, n),
        "transaction_memo": rng.choice(memos, n),
        "amount_usd": amount,
        "fee_usd": fee,
        "account_balance_before": balance_before,
        "account_balance_after": balance_after,
        "device_risk_score": risk_score,
        "network_latency_ms": latency,
        "transaction_frequency_1h": freq_1h,
        "transaction_frequency_24h": freq_24h,
        "is_foreign_transaction": is_foreign,
        "auth_factor_count": auth_factors,
        "target": target
    })

    out_path = output_dir / "large_dataset.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 3: Parquet Regression Dataset
# ==============================================================================
def generate_regression_parquet(output_dir: Path, seed: int = 42) -> Path:
    """Generate regression.parquet: ~5,000 rows, continuous price target."""
    rng = np.random.default_rng(seed)
    n = 5000

    sqft = rng.integers(600, 5500, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], n)
    lot_size = sqft + rng.integers(500, 25000, n)
    year_built = rng.integers(1960, 2024, n)
    distance = np.round(rng.uniform(1.0, 45.0, n), 2)
    prop_type = rng.choice(["Single Family", "Townhouse", "Condo", "Multi-Family"], n, p=[0.60, 0.20, 0.15, 0.05])
    rating = rng.integers(1, 11, n)
    garage = rng.choice([True, False], n, p=[0.75, 0.25])
    pool = rng.choice([True, False], n, p=[0.20, 0.80])

    # Continuous pricing equation + Gaussian noise
    price = (
        100000.0 +
        sqft * 180.0 +
        bedrooms * 12000.0 +
        bathrooms * 15000.0 +
        (year_built - 1960) * 800.0 -
        distance * 1500.0 +
        rating * 15000.0 +
        garage.astype(int) * 20000.0 +
        pool.astype(int) * 35000.0 +
        rng.normal(0, 25000.0, n)
    )
    price = np.round(np.clip(price, 80000.0, 2500000.0), 2)

    df = pl.DataFrame({
        "property_id": [f"PROP_{i+1:05d}" for i in range(n)],
        "square_feet": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size_sqft": lot_size,
        "year_built": year_built,
        "distance_to_center_km": distance,
        "property_type": prop_type,
        "neighborhood_rating": rating,
        "has_garage": garage,
        "has_pool": pool,
        "price": price
    })

    out_path = output_dir / "regression.parquet"
    df.write_parquet(out_path)
    return out_path


# ==============================================================================
# Scenario 4: Dataset with Missing Values (MCAR, MAR, All-Null)
# ==============================================================================
def generate_missing_values(output_dir: Path, seed: int = 42) -> Path:
    """Generate missing_values.csv: MCAR, MAR, high missingness, and 100% all-null column."""
    rng = np.random.default_rng(seed)
    n = 2000

    emp_type = rng.choice(["Full-time", "Part-time", "Self-employed", "Unemployed"], n, p=[0.65, 0.15, 0.12, 0.08])
    age_raw = rng.integers(18, 76, n).astype(float)
    inc_raw = np.round(rng.lognormal(10.8, 0.6, n), 2)
    dti_raw = np.round(rng.uniform(0.05, 0.85, n), 3)

    # MAR years_employed: 95% missing if unemployed, 5% otherwise
    years_raw = np.clip(rng.normal(7.0, 5.0, n), 0.0, 40.0)
    years_emp = [None if (emp == "Unemployed" and rng.random() < 0.95) or (emp != "Unemployed" and rng.random() < 0.05) else round(float(y), 1) for emp, y in zip(emp_type, years_raw)]

    # MAR credit_score: 75% missing if age < 23, 10% otherwise
    cred_raw = rng.integers(350, 850, n).astype(float)
    credit_score = [None if (a < 23 and rng.random() < 0.75) or (a >= 23 and rng.random() < 0.10) else float(c) for a, c in zip(age_raw, cred_raw)]

    # MCAR: age 10%, income 20%, dti 15%
    applicant_age = [None if rng.random() < 0.10 else float(a) for a in age_raw]
    annual_income = [None if rng.random() < 0.20 else float(inc) for inc in inc_raw]
    dti = [None if rng.random() < 0.15 else float(d) for d in dti_raw]

    # High missingness: 85% null
    guarantor = [None if rng.random() < 0.85 else rng.choice(["Yes", "No"]) for _ in range(n)]

    # 100% null column
    all_null: list[Any] = [None] * n

    defaulted = rng.choice([0, 1], n, p=[0.80, 0.20])

    df = pl.DataFrame({
        "loan_id": [f"LOAN_{i+1:05d}" for i in range(n)],
        "applicant_age": applicant_age,
        "annual_income": annual_income,
        "employment_type": emp_type,
        "years_employed": years_emp,
        "credit_score": credit_score,
        "debt_to_income_ratio": dti,
        "secondary_guarantor": guarantor,
        "all_null_feature": all_null,
        "defaulted": defaulted
    })

    out_path = output_dir / "missing_values.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 5: Dataset with Duplicate Records
# ==============================================================================
def generate_duplicates(output_dir: Path, seed: int = 42) -> Path:
    """Generate duplicates.csv: ~1,500 rows with exact row duplicates & ID collisions."""
    rng = np.random.default_rng(seed)
    n_base = 1000

    ids = [f"CUST_{i+1:05d}" for i in range(n_base)]
    plan = rng.choice(["Basic", "Pro", "Enterprise"], n_base)
    usage = np.round(rng.uniform(5.0, 500.0, n_base), 2)
    logins = rng.integers(1, 50, n_base)
    cycle = rng.choice(["Monthly", "Annual"], n_base)
    support = rng.integers(0, 10, n_base)
    retained = rng.choice([0, 1], n_base, p=[0.30, 0.70])

    df_base = pl.DataFrame({
        "customer_id": ids,
        "plan_type": plan,
        "monthly_usage_gb": usage,
        "login_count": logins,
        "billing_cycle": cycle,
        "support_calls": support,
        "retained": retained
    })

    # 300 exact row duplicates sampled with replacement
    idx_exact = rng.choice(n_base, 300, replace=True)
    df_exact = df_base[idx_exact]

    # 200 ID collision duplicates (same customer_id, updated non-key features)
    df_collisions = pl.DataFrame({
        "customer_id": ids[:200],
        "plan_type": rng.choice(["Basic", "Pro", "Enterprise"], 200),
        "monthly_usage_gb": np.round(rng.uniform(5.0, 500.0, 200), 2),
        "login_count": rng.integers(1, 50, 200),
        "billing_cycle": rng.choice(["Monthly", "Annual"], 200),
        "support_calls": rng.integers(0, 10, 200),
        "retained": rng.choice([0, 1], 200, p=[0.30, 0.70])
    })

    df_all = pl.concat([df_base, df_exact, df_collisions])
    perm = rng.permutation(df_all.height)
    df_shuffled = df_all[perm]

    out_path = output_dir / "duplicates.csv"
    df_shuffled.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 6: Dataset with Mixed / Dirty Data Types
# ==============================================================================
def generate_mixed_types(output_dir: Path, seed: int = 42) -> Path:
    """Generate mixed_types.csv: mixed ISO/US dates, dirty booleans, messy currency strings."""
    rng = np.random.default_rng(seed)
    n = 1000

    # Mixed dates
    iso_dates = [f"2023-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    us_dates = [f"{m:02d}/{d:02d}/2023" for m in range(1, 13) for d in range(1, 29)]
    date_entries = []
    for _ in range(n):
        r = rng.random()
        if r < 0.50:
            date_entries.append(rng.choice(iso_dates))
        elif r < 0.95:
            date_entries.append(rng.choice(us_dates))
        else:
            date_entries.append("invalid_date_entry")

    # Dirty booleans
    bool_choices = ["Y", "N", "yes", "no", "1", "0", "True", "False", "t", "f", "TRUE", "FALSE"]
    dirty_bools = rng.choice(bool_choices, n)

    # Messy currency with $ and commas
    amounts = np.round(rng.uniform(500.0, 50000.0, n), 2)
    dirty_currency = []
    for a in amounts:
        r = rng.random()
        if r < 0.45:
            dirty_currency.append(f"${a:,.2f}")
        elif r < 0.75:
            dirty_currency.append(f"{a:,.2f}")
        elif r < 0.95:
            dirty_currency.append(f"{a:.2f}")
        else:
            dirty_currency.append("unknown")

    # Messy percentages
    rates = np.round(rng.uniform(2.5, 25.0, n), 2)
    dirty_rates = []
    for rt in rates:
        r = rng.random()
        if r < 0.60:
            dirty_rates.append(f"{rt:.2f}%")
        elif r < 0.90:
            dirty_rates.append(f"{rt / 100.0:.4f}")
        else:
            dirty_rates.append(f"{rt:.1f} %")

    # Messy numbers with spaces and N/A strings
    scores = []
    for _ in range(n):
        r = rng.random()
        if r < 0.85:
            scores.append(f" {rng.integers(300, 850)} ")
        elif r < 0.95:
            scores.append("N/A")
        else:
            scores.append("None")

    approved = rng.choice([0, 1], n, p=[0.40, 0.60])

    df = pl.DataFrame({
        "application_id": [f"APP_{i+1:04d}" for i in range(n)],
        "application_date": date_entries,
        "is_employed": dirty_bools,
        "requested_amount": dirty_currency,
        "interest_rate": dirty_rates,
        "credit_score_raw": scores,
        "tier": rng.choice(["Bronze", "Silver", "Gold"], n),
        "approved": approved
    })

    out_path = output_dir / "mixed_types.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 7: Dataset with Severe Class Imbalance (99:1)
# ==============================================================================
def generate_imbalanced(output_dir: Path, seed: int = 42) -> Path:
    """Generate imbalanced.csv: exactly 99:1 ratio (1% positive class, 10,000 rows)."""
    rng = np.random.default_rng(seed)
    n = 10000
    n_pos = 100
    n_neg = n - n_pos

    # 9,900 Negative cases
    amt_neg = np.round(rng.lognormal(3.8, 0.8, n_neg), 2)
    dist_home_neg = np.round(rng.exponential(15.0, n_neg), 2)
    dist_txn_neg = np.round(rng.exponential(5.0, n_neg), 2)
    ratio_neg = np.round(rng.normal(1.0, 0.3, n_neg), 2)
    online_neg = rng.choice([0, 1], n_neg, p=[0.70, 0.30])
    y_neg = np.zeros(n_neg, dtype=int)

    # 100 Positive cases (fraud)
    amt_pos = np.array([round(float(rng.uniform(1.0, 5.0) if rng.random() < 0.3 else rng.uniform(1200.0, 8000.0)), 2) for _ in range(n_pos)])
    dist_home_pos = np.round(rng.uniform(80.0, 3000.0, n_pos), 2)
    dist_txn_pos = np.round(rng.uniform(50.0, 1500.0, n_pos), 2)
    ratio_pos = np.round(rng.uniform(4.0, 25.0, n_pos), 2)
    online_pos = rng.choice([0, 1], n_pos, p=[0.10, 0.90])
    y_pos = np.ones(n_pos, dtype=int)

    # Concatenate & Shuffle
    amt = np.concatenate([amt_neg, amt_pos])
    dist_home = np.concatenate([dist_home_neg, dist_home_pos])
    dist_txn = np.concatenate([dist_txn_neg, dist_txn_pos])
    ratio = np.concatenate([ratio_neg, ratio_pos])
    online = np.concatenate([online_neg, online_pos])
    target = np.concatenate([y_neg, y_pos])

    perm = rng.permutation(n)
    df = pl.DataFrame({
        "transaction_id": [f"TXN_{i+1:05d}" for i in range(n)],
        "amount": amt[perm],
        "distance_from_home_km": dist_home[perm],
        "distance_from_last_txn_km": dist_txn[perm],
        "ratio_to_median_price": ratio[perm],
        "online_order": online[perm],
        "is_fraud": target[perm]
    })

    out_path = output_dir / "imbalanced.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 8: Dataset with High-Cardinality Categorical Columns
# ==============================================================================
def generate_high_cardinality(output_dir: Path, seed: int = 42) -> Path:
    """Generate high_cardinality.csv: user IDs (cardinality 1.0), UUIDs, and ~850 ZIP codes."""
    rng = np.random.default_rng(seed)
    n = 5000

    user_ids = [f"USR_{i+1:05d}" for i in range(n)]

    # ~4800 unique UUIDs
    base_uuids = [str(uuid.UUID(int=int(rng.integers(0, 2**63)))) for _ in range(4800)]
    device_uuids = base_uuids + list(rng.choice(base_uuids, 200))
    rng.shuffle(device_uuids)

    # ~850 unique US ZIP codes
    base_zips = [f"{i:05d}" for i in rng.choice(range(10000, 99999), 850, replace=False)]
    zip_codes = rng.choice(base_zips, n)

    # ~2500 unique IP addresses
    base_ips = [f"192.168.{i//256}.{i%256}" for i in rng.choice(range(1, 65000), 2500, replace=False)]
    ip_addresses = rng.choice(base_ips, n)

    device_type = rng.choice(["mobile", "desktop", "tablet"], n, p=[0.65, 0.28, 0.07])
    pages_viewed = rng.integers(1, 20, n)
    time_spent = np.round(rng.exponential(120.0, n), 1)

    target = (rng.random(n) < (0.10 + 0.15 * (device_type == "desktop") + 0.02 * pages_viewed)).astype(int)

    df = pl.DataFrame({
        "user_id": user_ids,
        "device_uuid": device_uuids,
        "zip_code": zip_codes,
        "ip_address": ip_addresses,
        "device_type": device_type,
        "pages_viewed": pages_viewed,
        "time_on_page_sec": time_spent,
        "converted": target
    })

    out_path = output_dir / "high_cardinality.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 9: Dataset with Irrelevant and Constant Features
# ==============================================================================
def generate_irrelevant_constant(output_dir: Path, seed: int = 42) -> Path:
    """Generate irrelevant_constant.csv: zero-variance constant columns & random Gaussian noise."""
    rng = np.random.default_rng(seed)
    n = 1000

    c_num = np.full(n, 42.0)
    c_str = ["STATIC_CONFIG_CONSTANT"] * n
    near_c = ["ACTIVE"] * (n - 2) + ["INACTIVE", "INACTIVE"]

    noise_g1 = rng.normal(0, 1, n)
    noise_g2 = rng.normal(100, 25, n)
    noise_u = rng.uniform(-50, 50, n)

    sig1 = rng.standard_normal(n)
    sig2 = rng.standard_normal(n)
    sig3 = rng.integers(1, 10, n)

    z = 0.8 * sig1 - 1.2 * sig2 + 0.3 * (sig3 - 5)
    prob = 1.0 / (1.0 + np.exp(-z))
    target = (rng.random(n) < prob).astype(int)

    df = pl.DataFrame({
        "row_id": [f"ROW_{i+1:04d}" for i in range(n)],
        "constant_numeric": c_num,
        "constant_string": c_str,
        "near_constant": near_c,
        "noise_gaussian_1": noise_g1,
        "noise_gaussian_2": noise_g2,
        "noise_uniform": noise_u,
        "signal_feature_1": sig1,
        "signal_feature_2": sig2,
        "signal_feature_3": sig3,
        "target": target
    })

    out_path = output_dir / "irrelevant_constant.csv"
    df.write_csv(out_path)
    return out_path


# ==============================================================================
# Scenario 10: Dataset with Deliberately Invalid Target Configuration
# ==============================================================================
def generate_invalid_target(output_dir: Path, seed: int = 42) -> Path:
    """Generate invalid_target.csv (zero-variance target) + companion test variants."""
    rng = np.random.default_rng(seed)
    n = 1000

    # Primary: Zero-variance target
    df_zero_var = pl.DataFrame({
        "row_id": [f"ROW_{i+1:04d}" for i in range(n)],
        "feat_1": rng.standard_normal(n),
        "feat_2": rng.standard_normal(n),
        "feat_3": rng.integers(10, 100, n),
        "cat_1": rng.choice(["Category_A", "Category_B"], n),
        "target": np.zeros(n, dtype=int)  # Fatal: Zero variance
    })
    out_primary = output_dir / "invalid_target.csv"
    df_zero_var.write_csv(out_primary)

    # Companion variant A: Missing target column
    df_missing_target = pl.DataFrame({
        "row_id": [f"ROW_{i+1:04d}" for i in range(n)],
        "feat_1": rng.standard_normal(n),
        "feat_2": rng.standard_normal(n),
        "feat_3": rng.integers(10, 100, n),
        "cat_1": rng.choice(["Category_A", "Category_B"], n)
        # target completely missing
    })
    df_missing_target.write_csv(output_dir / "missing_target.csv")

    # Companion variant B: 100% Null target
    df_null_target = pl.DataFrame({
        "row_id": [f"ROW_{i+1:04d}" for i in range(n)],
        "feat_1": rng.standard_normal(n),
        "feat_2": rng.standard_normal(n),
        "cat_1": rng.choice(["Category_A", "Category_B"], n),
        "target": [None] * n  # Fatal: 100% null target
    })
    df_null_target.write_csv(output_dir / "null_target.csv")

    return out_primary


# ==============================================================================
# Master Generator and Schema Verification Harness
# ==============================================================================
GENERATORS = {
    "small_classification": generate_small_classification,
    "large_dataset": generate_large_dataset,
    "regression": generate_regression_parquet,
    "missing_values": generate_missing_values,
    "duplicates": generate_duplicates,
    "mixed_types": generate_mixed_types,
    "imbalanced": generate_imbalanced,
    "high_cardinality": generate_high_cardinality,
    "irrelevant_constant": generate_irrelevant_constant,
    "invalid_target": generate_invalid_target,
}


def verify_dataset_schema(dataset_key: str, path: Path) -> dict[str, Any]:
    """Verify structural integrity and scenario-specific invariants."""
    if not path.is_file():
        return {"status": "FAIL", "reason": f"File does not exist: {path}"}

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Read schema via Polars lazy scan
    if path.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path, infer_schema_length=1000, ignore_errors=True)

    schema = lf.collect_schema()
    col_names = schema.names()

    # Scenario-specific invariant assertions
    checks: list[tuple[bool, str]] = []

    if dataset_key == "small_classification":
        df = lf.collect()
        checks.append((df.height == 1000, f"Expected 1000 rows, got {df.height}"))
        checks.append((len(col_names) == 10, f"Expected 10 cols, got {len(col_names)}"))
        checks.append(("churn" in col_names, "Missing target 'churn'"))
        churn_mean = df["churn"].mean()
        checks.append((0.40 <= churn_mean <= 0.60, f"Expected balanced churn ~0.50, got {churn_mean:.3f}"))

    elif dataset_key == "large_dataset":
        row_count = lf.select(pl.len()).collect().item()
        checks.append((size_mb >= 50.0, f"Expected >= 50 MB, got {size_mb:.2f} MB"))
        checks.append((row_count >= 200_000, f"Expected >= 200,000 rows, got {row_count}"))
        checks.append((len(col_names) >= 15, f"Expected >= 15 cols, got {len(col_names)}"))

    elif dataset_key == "regression":
        df = lf.collect()
        checks.append((df.height == 5000, f"Expected 5000 rows, got {df.height}"))
        checks.append(("price" in col_names, "Missing continuous target 'price'"))
        checks.append((path.suffix.lower() == ".parquet", "Expected .parquet extension"))

    elif dataset_key == "missing_values":
        df = lf.collect()
        checks.append(("all_null_feature" in col_names, "Missing all_null_feature column"))
        checks.append((df["all_null_feature"].null_count() == df.height, "all_null_feature must be 100% null"))
        checks.append((df["applicant_age"].null_count() > 0, "applicant_age should have nulls"))

    elif dataset_key == "duplicates":
        df = lf.collect()
        exact_dups = df.height - df.unique().height
        checks.append((exact_dups >= 250, f"Expected >= 250 exact duplicates, got {exact_dups}"))

    elif dataset_key == "mixed_types":
        df = lf.collect()
        checks.append((df.height == 1000, f"Expected 1000 rows, got {df.height}"))
        checks.append(("is_employed" in col_names, "Missing is_employed"))
        checks.append(("requested_amount" in col_names, "Missing requested_amount"))

    elif dataset_key == "imbalanced":
        df = lf.collect()
        checks.append((df.height == 10000, f"Expected 10000 rows, got {df.height}"))
        pos_count = (df["is_fraud"] == 1).sum()
        checks.append((pos_count == 100, f"Expected exactly 100 positive fraud rows (1%), got {pos_count}"))

    elif dataset_key == "high_cardinality":
        df = lf.collect()
        checks.append((df["user_id"].n_unique() == df.height, "user_id should be unique per row"))
        checks.append((df["zip_code"].n_unique() >= 500, "zip_code should have >= 500 unique values"))

    elif dataset_key == "irrelevant_constant":
        df = lf.collect()
        checks.append((df["constant_numeric"].var() == 0.0, "constant_numeric variance must be 0.0"))
        checks.append((df["constant_string"].n_unique() == 1, "constant_string must have 1 unique value"))

    elif dataset_key == "invalid_target":
        df = lf.collect()
        checks.append((df["target"].n_unique() <= 1, "target must have <= 1 unique value (zero variance)"))

    failed_checks = [msg for passed, msg in checks if not passed]
    if failed_checks:
        return {"status": "FAIL", "reason": "; ".join(failed_checks), "size_mb": size_mb}
    return {"status": "PASS", "size_mb": size_mb, "col_count": len(col_names)}


def generate_all_datasets(
    output_dir: Path = Path("data/test_fixtures"),
    copy_to_input: bool = False,
    input_dir: Path = Path("data/input"),
    seed: int = 42,
    large_rows: int = 220_000,
    verify: bool = True
) -> dict[str, Path]:
    """Programmatic entry point to generate all 10 synthetic datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_to_input:
        input_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for name, gen_func in GENERATORS.items():
        if name == "large_dataset":
            path = gen_func(output_dir, row_count=large_rows, seed=seed)
        else:
            path = gen_func(output_dir, seed=seed)
        results[name] = path

        if copy_to_input:
            target_path = input_dir / path.name
            shutil.copy2(path, target_path)

    # Also copy companion files if generated
    if copy_to_input:
        for extra in ["missing_target.csv", "null_target.csv"]:
            extra_src = output_dir / extra
            if extra_src.is_file():
                shutil.copy2(extra_src, input_dir / extra)

    if verify:
        all_passed = True
        print("\n" + "=" * 80)
        print(f"{'DATASET KEY':<24} | {'FILENAME':<24} | {'SIZE (MB)':<10} | {'STATUS'}")
        print("-" * 80)
        for key, path in results.items():
            check = verify_dataset_schema(key, path)
            status = check["status"]
            size_mb = check.get("size_mb", 0.0)
            if status != "PASS":
                all_passed = False
                print(f"{key:<24} | {path.name:<24} | {size_mb:<10.2f} | FAIL: {check.get('reason')}")
            else:
                print(f"{key:<24} | {path.name:<24} | {size_mb:<10.2f} | PASS")
        print("=" * 80 + "\n")
        if not all_passed:
            raise RuntimeError("One or more synthetic test datasets failed verification checks.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Synthetic Test Dataset Generator")
    parser.add_argument("--all", action="store_true", help="Generate all datasets (shorthand for --dataset all)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/test_fixtures"), help="Target fixture directory")
    parser.add_argument("--copy-to-input", action="store_true", help="Copy generated datasets to data/input/")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"), help="Input directory for main CLI discovery")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--large-rows", type=int, default=220_000, help="Row count for large_dataset.csv (>= 200,000)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + list(GENERATORS.keys()), help="Dataset scenario to generate")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify dataset invariants post-generation")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_to_input:
        args.input_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.dataset == "all":
        generate_all_datasets(
            output_dir=args.output_dir,
            copy_to_input=args.copy_to_input,
            input_dir=args.input_dir,
            seed=args.seed,
            large_rows=args.large_rows,
            verify=args.verify
        )
    else:
        gen_func = GENERATORS[args.dataset]
        if args.dataset == "large_dataset":
            path = gen_func(args.output_dir, row_count=args.large_rows, seed=args.seed)
        else:
            path = gen_func(args.output_dir, seed=args.seed)

        if args.copy_to_input:
            shutil.copy2(path, args.input_dir / path.name)
            if args.dataset == "invalid_target":
                for extra in ["missing_target.csv", "null_target.csv"]:
                    extra_src = args.output_dir / extra
                    if extra_src.is_file():
                        shutil.copy2(extra_src, args.input_dir / extra)

        if args.verify:
            check = verify_dataset_schema(args.dataset, path)
            print(f"Dataset {args.dataset}: Status={check['status']}, Size={check.get('size_mb', 0):.2f} MB")
            if check["status"] != "PASS":
                sys.exit(1)


if __name__ == "__main__":
    main()
