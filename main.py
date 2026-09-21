"""Run the whole pipeline end to end:  python main.py"""
from src.data_quality import run_quality_checks
from src.eda import run_eda
from src.generate_data import generate_loan_data
from src.config import RAW_FILE
from src.train_model import run_modelling


def main() -> None:
    print("STEP 1/4  Generating data ...")
    raw = generate_loan_data()
    raw.to_csv(RAW_FILE, index=False)
    print(f"Saved {len(raw)} rows to {RAW_FILE}")

    print("\nSTEP 2/4  Checking data quality and cleaning ...")
    clean = run_quality_checks(raw)

    print("\nSTEP 3/4  Exploratory data analysis ...")
    run_eda(clean)

    print("\nSTEP 4/4  Modelling and evaluation ...")
    run_modelling(clean)

    print("\nDone. See the reports/ folder for all outputs.")


if __name__ == "__main__":
    main()
