from typing import List, Tuple

import pandas as pd

TARGET = "is_fraud"
ID_COLUMN = "transaction_id"

FEATURE_COLUMNS: List[str] = [
    "amount",
    "hour",
    "day_of_week",
    "merchant_risk_score",
    "customer_age",
    "account_age_days",
    "previous_failed_attempts",
    "num_transactions_24h",
    "foreign_transaction",
    "card_present",
    "device_trust_score",
    "is_weekend",
    "night_transaction",
]


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int).copy()
