import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, split_features_target
from src.metrics import classification_metrics


def train(input_path: str, model_dir: str = "models", threshold: float = 0.35) -> dict:
    df = pd.read_csv(input_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    latency_ms_per_1000 = (time.perf_counter() - start_pred) * 1000 / (len(X_test) / 1000)

    metrics = classification_metrics(y_test, y_prob, threshold=threshold)
    metrics["training_seconds"] = training_seconds
    metrics["latency_ms_per_1000_transactions"] = latency_ms_per_1000
    metrics["test_rows"] = int(len(X_test))
    metrics["fraud_rate"] = float(y.mean())

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "threshold": threshold}, model_path / "fraud_model.joblib")
    X_test.assign(is_fraud=y_test.values, fraud_probability=y_prob).to_csv(model_path / "test_predictions.csv", index=False)
    with open(model_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/transactions.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    metrics = train(args.input, args.model_dir, args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
