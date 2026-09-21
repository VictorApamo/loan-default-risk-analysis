"""Step 4 & 5 - Modelling and evaluation for loan default prediction.

Workflow
1. Feature engineering (loan_to_income ratio).
2. Stratified train/test split.
3. Compare two models with 5-fold cross-validation on TRAINING data.
4. Choose the best model by ROC-AUC (independent of any threshold).
5. Pick a decision threshold that minimises business cost, using out-of-fold
   training predictions (so the test set is never used to tune anything).
6. Evaluate once on the test set at the default 0.5 threshold and the cost-based one.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, CV_FOLDS, FIGURES_DIR, ID_COL,
                        RANDOM_STATE, REPORTS_DIR, TARGET, TEST_SIZE)

CATEGORICAL = ["home_ownership", "loan_purpose"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: how large is the loan relative to the borrower's income?"""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    return df


def build_models(numeric_cols: list) -> dict:
    def preprocessor():
        return ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])

    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                             random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def expected_cost(y_true, proba, threshold: float) -> float:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "total_cost": float(expected_cost(y_true, proba, threshold)),
    }


def run_modelling(df: pd.DataFrame) -> dict:
    df = add_features(df)
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # ---- Model comparison (cross-validation, training data only) ----
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
               "precision": "precision", "recall": "recall"}
    models, rows = build_models(numeric_cols), []
    for name, pipe in models.items():
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({"model": name, **{m: round(s[f"test_{m}"].mean(), 4) for m in scoring}})
    cv_table = pd.DataFrame(rows).set_index("model")
    cv_table.to_csv(REPORTS_DIR / "cv_model_comparison.csv")
    print("\nCross-validation results (training data):\n", cv_table)

    best_name = cv_table["roc_auc"].idxmax()
    best = models[best_name]

    # ---- Choose threshold from out-of-fold predictions (no test-set peeking) ----
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_train, oof, t) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmin(costs))])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, costs, color="#2E86AB")
    ax.axvline(best_threshold, color="#D1495B", linestyle="--",
               label=f"Chosen threshold = {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold (predicted default probability)")
    ax.set_ylabel("Total cost (training out-of-fold)")
    ax.set_title(f"Cost vs threshold (FN cost = {COST_FALSE_NEGATIVE:g}x FP cost)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_vs_threshold.png", dpi=150)
    plt.close(fig)

    # ---- Final fit and one-time test evaluation ----
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    results = {
        "best_model": best_name,
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "baseline_default_rate": round(float(y_test.mean()), 4),
        "baseline_cost_approve_everyone": float(y_test.sum() * COST_FALSE_NEGATIVE),
        "at_default_threshold_0.5": evaluate(y_test, proba, 0.5),
        "at_cost_based_threshold": evaluate(y_test, proba, best_threshold),
        "cost_assumption": f"FN = {COST_FALSE_NEGATIVE:g}, FP = {COST_FALSE_POSITIVE:g}",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nFinal test-set results:", json.dumps(results, indent=2))

    # ---- Figures ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC curve (test set)")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name=best_name)
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.5, label="No-skill baseline")
    axes[1].legend()
    axes[1].set_title("Precision-recall curve (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_and_pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= best_threshold).astype(int),
        display_labels=["Repaid", "Default"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix (threshold {best_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Feature importance via the fitted model
    names = best.named_steps["prep"].get_feature_names_out()
    inner = best.named_steps["model"]
    imp = inner.feature_importances_ if hasattr(inner, "feature_importances_") else abs(inner.coef_[0])
    top = pd.Series(imp, index=[n.split("__")[-1] for n in names]).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values, color="#2E86AB")
    ax.set_title("Top 10 most influential features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    return results
