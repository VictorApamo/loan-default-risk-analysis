"""Step 3 - Exploratory data analysis: how does default risk vary across borrowers?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET


def run_eda(df: pd.DataFrame) -> None:
    df = df.copy()

    # 1. Descriptive statistics
    df.describe().T.round(2).to_csv(REPORTS_DIR / "descriptive_statistics.csv")

    # 2. Default rate by credit-score band
    df["score_band"] = pd.cut(df["credit_score"], [299, 580, 670, 740, 800, 850],
                              labels=["Poor\n<580", "Fair\n580-669", "Good\n670-739",
                                      "Very good\n740-799", "Excellent\n800+"])
    by_band = df.groupby("score_band", observed=True)[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_band.index.astype(str), by_band.values, color="#2E86AB")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by credit-score band")
    for i, v in enumerate(by_band.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_credit_score.png", dpi=150)
    plt.close(fig)

    # 3. Default rate by loan purpose and home ownership
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["loan_purpose", "home_ownership"]):
        rate = (df.groupby(col)[TARGET].mean() * 100).sort_values()
        ax.barh(rate.index, rate.values, color="#D1495B")
        ax.set_xlabel("Default rate (%)")
        ax.set_title(f"Default rate by {col.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_by_category.png", dpi=150)
    plt.close(fig)

    # 4. Distribution of debt-to-income for defaulters vs non-defaulters
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df.loc[df[TARGET] == 0, "debt_to_income"], bins=30, alpha=0.6, density=True,
            label="Repaid", color="#4C9F70")
    ax.hist(df.loc[df[TARGET] == 1, "debt_to_income"], bins=30, alpha=0.6, density=True,
            label="Defaulted", color="#D1495B")
    ax.set_xlabel("Debt-to-income ratio")
    ax.set_ylabel("Density")
    ax.set_title("Debt-to-income: repaid vs defaulted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "debt_to_income_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Correlation of numeric features with default
    corr = df.select_dtypes("number").drop(columns=["applicant_id"]).corr()[TARGET].drop(TARGET)
    corr = corr.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(corr.index, corr.values, color=["#D1495B" if v > 0 else "#4C9F70" for v in corr.values])
    ax.set_title("Correlation of each numeric feature with default")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_with_default.png", dpi=150)
    plt.close(fig)

    print("EDA complete: figures saved to reports/figures/, statistics saved to reports/")
