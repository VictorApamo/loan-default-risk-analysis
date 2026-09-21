"""Small sanity tests.  Run with:  pytest"""
from src.config import TARGET
from src.data_quality import validate_business_rules
from src.generate_data import generate_loan_data
from src.train_model import add_features


def test_generation_is_reproducible():
    a, b = generate_loan_data(500, seed=1), generate_loan_data(500, seed=1)
    assert a.equals(b)


def test_default_rate_is_realistic():
    df = generate_loan_data(5000)
    assert 0.08 < df[TARGET].mean() < 0.30


def test_injected_quality_problems_exist():
    df = generate_loan_data(1000)
    assert df.isna().sum().sum() > 0
    assert df.duplicated().sum() == 25


def test_business_rules_hold():
    df = generate_loan_data(1000)
    assert sum(validate_business_rules(df).values()) == 0


def test_feature_engineering():
    df = add_features(generate_loan_data(200))
    assert "loan_to_income" in df.columns
