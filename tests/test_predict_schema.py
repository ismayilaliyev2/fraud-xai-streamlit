from src.features import FEATURE_COLUMNS


def test_feature_columns_are_unique():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert "amount" in FEATURE_COLUMNS
    assert "merchant_risk_score" in FEATURE_COLUMNS
