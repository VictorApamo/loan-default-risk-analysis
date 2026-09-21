"""Step 2 - Data preparation: quality checks, business-rule validation and cleaning.

Missing values are REPORTED here but imputed later inside the modelling pipeline,
so the imputation values are learned from training data only (no data leakage).
"""
import pandas as pd

from src.config import CLEAN_FILE, RAW_FILE, REPORTS_DIR, TARGET


def validate_business_rules(df: pd.DataFrame) -> dict:
    """Return the number of rows that break each logical rule."""
    return {
        "age below 18": int((df["age"] < 18).sum()),
        "credit_score outside 300-850": int((~df["credit_score"].between(300, 850)).sum()),
        "non-positive annual_income": int((df["annual_income"] <= 0).sum()),
        "employment_years greater than age - 16": int((df["employment_years"] > df["age"] - 16).sum()),
        "debt_to_income outside 0-1": int((~df["debt_to_income"].between(0, 1)).sum()),
    }


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    lines = ["DATA QUALITY REPORT", "=" * 60]
    lines.append(f"Rows: {len(df)} | Columns: {df.shape[1]}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    lines.append("\nMissing values:")
    for col, n in missing.items():
        lines.append(f"  - {col}: {n} ({n / len(df):.1%})")
    lines.append("  Decision: impute with the median inside the modelling pipeline.")

    dup = int(df.duplicated().sum())
    lines.append(f"\nExact duplicate rows: {dup}  -> removed")

    lines.append("\nBusiness-rule violations:")
    for rule, n in validate_business_rules(df).items():
        lines.append(f"  - {rule}: {n}")

    lines.append(f"\nDefault rate (target balance): {df[TARGET].mean():.1%}")
    lines.append("  Decision: classes are imbalanced, so judge models by ROC-AUC/PR-AUC (not accuracy) and tune the decision threshold to business cost.")

    clean = df.drop_duplicates().reset_index(drop=True)
    lines.append(f"\nRows after cleaning: {len(clean)}")

    (REPORTS_DIR / "data_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")
    clean.to_csv(CLEAN_FILE, index=False)
    print("\n".join(lines))
    return clean


if __name__ == "__main__":
    run_quality_checks(pd.read_csv(RAW_FILE))
