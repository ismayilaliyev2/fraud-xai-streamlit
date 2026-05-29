# Model Card: Fraud Detection with Explainable AI

## Model Overview

This project uses an XGBoost binary classifier to estimate the probability that a financial transaction is fraudulent. The model is designed for portfolio demonstration, explainable AI practice, and fraud analytics prototyping.

## Intended Use

The model is intended to support fraud review prioritisation. It should not be used as the only decision-making mechanism for blocking customer transactions.

## Dataset

The default dataset is synthetically generated using `src/data_generator.py`. It simulates transaction amount, time, merchant risk, account age, device trust, failed attempts, transaction velocity, card-present status, and foreign transaction indicators.

## Inputs

- Transaction amount
- Transaction hour and day of week
- Merchant risk score
- Customer age
- Account age in days
- Previous failed attempts
- Number of transactions in last 24 hours
- Foreign transaction flag
- Card-present flag
- Device trust score
- Weekend and night transaction indicators

## Output

- Fraud probability
- Binary risk decision based on a configurable threshold
- SHAP-based feature explanations

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix values
- Prediction latency per 1,000 transactions

Accuracy is not treated as the main metric because fraud detection usually involves class imbalance. F1, recall, precision, ROC-AUC, and PR-AUC are more useful.

## Limitations

- Synthetic data cannot fully represent real fraud behaviour.
- Real fraud patterns change over time due to adversarial behaviour.
- The model does not include graph-based fraud rings, identity links, or merchant networks.
- The system does not currently perform drift detection.
- The dashboard is a prototype and not a regulated production fraud system.

## Bias and Fairness Risks

The model includes customer age as a demonstration feature. In a real financial setting, age and other sensitive or proxy-sensitive attributes should be reviewed carefully for fairness, compliance, and explainability. Decisions should be audited to ensure that protected or vulnerable groups are not unfairly affected.

## Risk and Governance Assumptions

- A human fraud analyst reviews high-risk transactions.
- Thresholds are tuned based on business cost trade-offs.
- False positives create customer friction.
- False negatives create fraud loss.
- Model performance should be monitored after deployment.

## Recommended Improvements

- Replace synthetic data with a real approved fraud dataset.
- Add data drift and concept drift monitoring.
- Add model registry and experiment tracking with MLflow.
- Add threshold optimisation based on business cost.
- Add fairness analysis and model governance reports.
