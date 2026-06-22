# Yogyank Safe Baseline
## Start Time (IST)
09:00 AM IST
## End Time (IST)
10:25 AM IST
## Approximate Time Spent
85 Minutes
---

## Setup
Install dependencies:
pip install pandas numpy scikit-learn xgboost joblib

Run:
python fixed_yogyank_training.py
---

## Files Generated
artifacts/
* yogyank_pipeline.pkl
* validation_metrics.json
* feature_list.json
* model_metadata.json
---

## Completed

Leakage review
Temporal validation
pipeline-based preprocessing
Missing value handling
Artifact generation
Metadata tracking
Reproducible training
---

## Skipped Due To Time
* Hyperparameter optimization
* SHAP explainability
* Drift monitoring
* Automated testing
---

## Assumptions
The following assumptions were made:
1. application_year is available before scoring.
2. defaulted_in_next_12_months is unavailable at scoring time.
3. PM-Kisan status is available before scoring.
4. Historical repayment score is available before scoring.
Because feature definitions were not provided, these assumptions require business validation.

---

## Validation Approach
Temporal validation was used.

Training:
All years before the most recent year.

Validation:
Most recent year.
This better reflects future scoring compared to a random split.

---

## Do I Trust The Result?
Partially.
I trust the result more than the original implementation because leakage has been reduced and validation is more realistic.
However, I would not approve production deployment until feature availability and data lineage are confirmed.
