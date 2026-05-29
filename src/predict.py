from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/fraud_model.joblib")


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train --input data/raw/transactions.csv --model-dir models"
        )
    return joblib.load(model_path)


def predict_one(payload: Dict, model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict:
    bundle = load_model(model_path)
    features: List[str] = bundle["features"]
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.35)
    row = pd.DataFrame([payload])[features]
    probability = float(model.predict_proba(row)[:, 1][0])
    return {
        "fraud_probability": probability,
        "prediction": int(probability >= threshold),
        "risk_label": "High risk" if probability >= threshold else "Low risk",
        "threshold": threshold,
    }
