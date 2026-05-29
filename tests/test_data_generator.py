from src.data_generator import generate_transactions


def test_data_generator_has_target_and_rows():
    df = generate_transactions(rows=1000, seed=1)
    assert len(df) == 1000
    assert "is_fraud" in df.columns
    assert df["is_fraud"].isin([0, 1]).all()
