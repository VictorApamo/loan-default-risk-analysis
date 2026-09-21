"""Step 1 - Data collection: build a reproducible SYNTHETIC loan-application dataset.

Why synthetic? Real lending data is confidential. Generating it from stated rules
gives a realistic, shareable dataset and lets us know the "true" drivers of default.
Deliberate data-quality problems (missing values, duplicates) are injected so the
data-quality step has something real to find.

Swap in a real dataset (e.g. Kaggle "Give Me Some Credit") by saving it as
data/raw/loan_applications_raw.csv with the same column names.
"""
import numpy as np
import pandas as pd

from src.config import N_APPLICANTS, RANDOM_STATE, RAW_FILE, TARGET


def generate_loan_data(n: int = N_APPLICANTS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 71, n)
    annual_income = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n), -2)
    employment_years = np.minimum(np.clip(rng.normal((age - 18) * 0.4, 3), 0, None), age - 18).round(1)
    credit_score = np.clip(rng.normal(670, 80, n), 300, 850).round().astype(int)
    loan_amount = np.maximum(np.round(annual_income * rng.uniform(0.05, 0.9, n), -2), 500)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.20, 0.35, 0.15, 0.20])
    num_credit_lines = rng.poisson(4, n)
    debt_to_income = np.round(rng.beta(2, 5, n) * 0.9, 3)
    previous_defaults = rng.choice([0, 1, 2], n, p=[0.88, 0.09, 0.03])
    home_ownership = rng.choice(["Rent", "Mortgage", "Own"], n, p=[0.45, 0.40, 0.15])
    loan_purpose = rng.choice(
        ["Debt consolidation", "Home improvement", "Education", "Business", "Medical", "Car"],
        n, p=[0.30, 0.15, 0.15, 0.15, 0.10, 0.15],
    )

    # "True" data-generating rule: default risk rises with high debt, large loans relative
    # to income, past defaults and low credit scores; it falls with job stability.
    z = (
        -2.6
        - 0.012 * (credit_score - 670)
        + 3.5 * (debt_to_income - 0.25)
        + 1.5 * (loan_amount / annual_income)
        - 0.06 * employment_years
        + 0.7 * previous_defaults
        + 0.35 * (home_ownership == "Rent")
        + 0.006 * (loan_term_months - 36)
        + 0.25 * (loan_purpose == "Business")
        + rng.normal(0, 0.5, n)          # unexplained randomness
    )
    default = rng.binomial(1, 1 / (1 + np.exp(-z)))

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "num_credit_lines": num_credit_lines,
        "debt_to_income": debt_to_income,
        "previous_defaults": previous_defaults,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        TARGET: default,
    })

    # ---- Inject realistic data-quality problems ----
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "annual_income"] = np.nan
    df.loc[rng.choice(n, int(0.04 * n), replace=False), "employment_years"] = np.nan
    df = pd.concat([df, df.sample(25, random_state=seed)], ignore_index=True)  # 25 duplicate rows
    return df


if __name__ == "__main__":
    data = generate_loan_data()
    data.to_csv(RAW_FILE, index=False)
    print(f"Generated {len(data)} rows -> {RAW_FILE} | default rate: {data[TARGET].mean():.1%}")
