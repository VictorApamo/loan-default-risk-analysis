"""Central configuration so every step uses the same paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "loan_applications_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "loan_applications_clean.csv"

TARGET = "default"            # 1 = borrower defaulted, 0 = repaid
ID_COL = "applicant_id"
RANDOM_STATE = 42             # fixed seed -> reproducible results
N_APPLICANTS = 5000
TEST_SIZE = 0.20
CV_FOLDS = 5

# Business assumption used for threshold selection (cost units, not dollars):
# approving a borrower who defaults costs 5x more than rejecting a good one.
COST_FALSE_NEGATIVE = 5.0     # approved but defaulted (lost principal)
COST_FALSE_POSITIVE = 1.0     # rejected but would have repaid (lost interest)

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
