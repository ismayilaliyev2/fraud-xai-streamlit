# Data

This project includes a reproducible synthetic fraud transaction data generator in `src/data_generator.py`.

Why synthetic data?
- The project can run without external credentials.
- It avoids exposing sensitive financial records.
- It still simulates realistic fraud patterns such as unusual transaction amount, risky merchant category, night activity, foreign transaction, failed attempts, and account age.

To generate data:

```bash
python -m src.data_generator --rows 50000 --output data/raw/transactions.csv
```
