import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=3.2, sigma=1.0, size=rows).round(2)
    hour = rng.integers(0, 24, size=rows)
    day_of_week = rng.integers(0, 7, size=rows)
    merchant_risk_score = rng.beta(2, 5, size=rows)
    customer_age = rng.integers(18, 80, size=rows)
    account_age_days = rng.exponential(scale=550, size=rows).clip(1, 3650).round().astype(int)
    previous_failed_attempts = rng.poisson(lam=0.3, size=rows).clip(0, 6)
    num_transactions_24h = rng.poisson(lam=2.2, size=rows).clip(0, 20)
    foreign_transaction = rng.binomial(1, 0.08, size=rows)
    card_present = rng.binomial(1, 0.72, size=rows)
    device_trust_score = rng.beta(5, 2, size=rows)
    is_weekend = np.isin(day_of_week, [5, 6]).astype(int)
    night_transaction = ((hour <= 5) | (hour >= 23)).astype(int)

    amount_z = (np.log1p(amount) - np.log1p(amount).mean()) / np.log1p(amount).std()
    risk_logit = (
        -4.4
        + 0.75 * amount_z
        + 1.45 * merchant_risk_score
        + 0.7 * night_transaction
        + 0.9 * foreign_transaction
        + 0.35 * previous_failed_attempts
        + 0.07 * num_transactions_24h
        - 1.2 * device_trust_score
        - 0.00035 * account_age_days
        - 0.45 * card_present
    )
    probability = 1 / (1 + np.exp(-risk_logit))
    is_fraud = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:07d}" for i in range(rows)],
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "merchant_risk_score": merchant_risk_score.round(4),
            "customer_age": customer_age,
            "account_age_days": account_age_days,
            "previous_failed_attempts": previous_failed_attempts,
            "num_transactions_24h": num_transactions_24h,
            "foreign_transaction": foreign_transaction,
            "card_present": card_present,
            "device_trust_score": device_trust_score.round(4),
            "is_weekend": is_weekend,
            "night_transaction": night_transaction,
            "is_fraud": is_fraud,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions(args.rows, args.seed)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
