from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_one

app = FastAPI(title="Fraud Detection API", version="1.0.0")
MODEL_PATH = Path("models/fraud_model.joblib")


class Transaction(BaseModel):
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    customer_age: int = Field(..., ge=18, le=100)
    account_age_days: int = Field(..., ge=1)
    previous_failed_attempts: int = Field(..., ge=0)
    num_transactions_24h: int = Field(..., ge=0)
    foreign_transaction: int = Field(..., ge=0, le=1)
    card_present: int = Field(..., ge=0, le=1)
    device_trust_score: float = Field(..., ge=0, le=1)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)
    night_transaction: Optional[int] = Field(None, ge=0, le=1)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.post("/predict")
def predict(transaction: Transaction):
    payload = transaction.model_dump()
    payload["is_weekend"] = int(payload["day_of_week"] in [5, 6]) if payload["is_weekend"] is None else payload["is_weekend"]
    payload["night_transaction"] = int(payload["hour"] <= 5 or payload["hour"] >= 23) if payload["night_transaction"] is None else payload["night_transaction"]
    return predict_one(payload, MODEL_PATH)
