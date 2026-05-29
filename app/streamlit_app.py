import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.features import FEATURE_COLUMNS

st.set_page_config(page_title="Fraud Detection XAI", page_icon="🛡️", layout="wide")

MODEL_PATH = Path("models/fraud_model.joblib")
METRICS_PATH = Path("models/metrics.json")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_json(METRICS_PATH, typ="series").to_dict()
    return {}

st.title("🛡️ Fraud Detection with Explainable AI")
st.caption("XGBoost model + SHAP explanations + business-oriented fraud risk dashboard")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.data_generator` and `python -m src.train` first.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
threshold = bundle.get("threshold", 0.35)
metrics = load_metrics()

with st.sidebar:
    st.header("Transaction Input")
    amount = st.number_input("Amount", min_value=0.0, value=120.0, step=10.0)
    hour = st.slider("Hour", 0, 23, 22)
    day_of_week = st.slider("Day of week", 0, 6, 5, help="0=Monday, 6=Sunday")
    merchant_risk_score = st.slider("Merchant risk score", 0.0, 1.0, 0.65)
    customer_age = st.slider("Customer age", 18, 90, 34)
    account_age_days = st.number_input("Account age days", min_value=1, value=220)
    previous_failed_attempts = st.slider("Previous failed attempts", 0, 6, 1)
    num_transactions_24h = st.slider("Transactions in last 24h", 0, 20, 5)
    foreign_transaction = st.selectbox("Foreign transaction", [0, 1], index=1)
    card_present = st.selectbox("Card present", [0, 1], index=0)
    device_trust_score = st.slider("Device trust score", 0.0, 1.0, 0.35)

payload = {
    "amount": amount,
    "hour": hour,
    "day_of_week": day_of_week,
    "merchant_risk_score": merchant_risk_score,
    "customer_age": customer_age,
    "account_age_days": account_age_days,
    "previous_failed_attempts": previous_failed_attempts,
    "num_transactions_24h": num_transactions_24h,
    "foreign_transaction": foreign_transaction,
    "card_present": card_present,
    "device_trust_score": device_trust_score,
    "is_weekend": int(day_of_week in [5, 6]),
    "night_transaction": int(hour <= 5 or hour >= 23),
}

row = pd.DataFrame([payload])[FEATURE_COLUMNS]
probability = float(model.predict_proba(row)[0, 1])
prediction = int(probability >= threshold)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Probability", f"{probability:.1%}")
col2.metric("Decision", "Review" if prediction else "Approve")
col3.metric("Threshold", f"{threshold:.0%}")
col4.metric("Model F1", f"{metrics.get('f1', 0):.3f}")

st.subheader("Business Interpretation")
if prediction:
    st.warning("This transaction should be reviewed before approval because the model estimates high fraud risk.")
else:
    st.success("This transaction appears low risk under the current decision threshold.")

st.subheader("Explainability: SHAP Feature Contribution")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(row)
contrib = pd.DataFrame({"feature": FEATURE_COLUMNS, "shap_value": shap_values[0], "value": row.iloc[0].values})
contrib["impact"] = contrib["shap_value"].abs()
contrib = contrib.sort_values("impact", ascending=False).head(10)
fig = px.bar(contrib, x="shap_value", y="feature", orientation="h", hover_data=["value"], title="Top local drivers of this prediction")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Metrics")
if metrics:
    metric_df = pd.DataFrame([metrics]).T.reset_index()
    metric_df.columns = ["metric", "value"]
    st.dataframe(metric_df, use_container_width=True)
else:
    st.info("Metrics are created after training and saved to models/metrics.json")
