# Methodology (CRISP-DM)

| Phase | What was done | Where |
|---|---|---|
| **1. Business understanding** | Framed the task as credit-risk classification. Recognised that errors have unequal costs and defined a cost matrix (false negative = 5, false positive = 1). | `README.md` section 1, `src/config.py` |
| **2. Data understanding** | Generated a synthetic applicant dataset from a documented risk rule, reviewed descriptive statistics and default rates by segment. | `src/generate_data.py`, `src/eda.py` |
| **3. Data preparation** | Removed exact duplicates, validated business rules, reported missing values. Engineered `loan_to_income`. Imputation, scaling and encoding are done *inside* the modelling pipeline to prevent data leakage. | `src/data_quality.py`, `src/train_model.py` |
| **4. Modelling** | Built Logistic Regression and Random Forest pipelines. Stratified 80/20 train/test split. | `src/train_model.py` |
| **5. Evaluation** | Compared models by 5-fold cross-validation (ROC-AUC, PR-AUC, precision, recall). Chose the decision threshold from out-of-fold training predictions by minimising expected cost. Evaluated once on the test set. | `src/train_model.py`, `reports/` |
| **6. Deployment** | Out of scope. A real deployment would need monitoring for data drift, a fairness audit and regulatory review. | n/a |

## Key methodological decisions

1. **Synthetic data with a fixed seed:** removes confidentiality issues and makes results reproducible.
2. **Injected quality problems:** lets the project demonstrate detection and treatment, not just modelling.
3. **Imputation inside the pipeline:** median values are learned from training folds only.
4. **Metric choice:** the data is imbalanced (about 16% defaults), so accuracy is rejected in favour of ROC-AUC, PR-AUC and cost.
5. **Threshold chosen without the test set:** out-of-fold predictions on training data pick the cut-off, keeping the test set an honest final check.
6. **Cost-sensitive decision rule:** turns a probability model into a business decision.
