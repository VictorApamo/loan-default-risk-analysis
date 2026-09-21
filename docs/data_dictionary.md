# Data Dictionary

**Source:** synthetic (`src/generate_data.py`, seed 42)
**Rows:** 5,025 raw (5,000 after removing 25 duplicates) | **Columns:** 13

| Column | Type | Description | Notes |
|---|---|---|---|
| `applicant_id` | integer | Unique applicant identifier | Dropped before modelling |
| `age` | integer | Applicant age in years | 18 to 70 |
| `annual_income` | float | Gross annual income (currency units) | About 3% missing |
| `employment_years` | float | Years in current employment | About 4% missing |
| `credit_score` | integer | Credit score | 300 to 850 |
| `loan_amount` | float | Requested loan amount | Minimum 500 |
| `loan_term_months` | integer | Loan length in months | 12, 24, 36, 48 or 60 |
| `num_credit_lines` | integer | Number of existing credit lines | Poisson-distributed |
| `debt_to_income` | float | Monthly debt payments divided by monthly income | 0 to 0.9 |
| `previous_defaults` | integer | Number of prior loan defaults | 0, 1 or 2 |
| `home_ownership` | categorical | Rent, Mortgage or Own | |
| `loan_purpose` | categorical | Debt consolidation, Home improvement, Education, Business, Medical or Car | |
| `default` | binary integer | **Target.** 1 = borrower defaulted, 0 = repaid | About 16% positive |

## Engineered feature (created in `src/train_model.py`)

| Column | Formula | Meaning |
|---|---|---|
| `loan_to_income` | `loan_amount / annual_income` | How large the loan is relative to the borrower's yearly income |
