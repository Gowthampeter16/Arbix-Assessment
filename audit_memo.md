# Audit Memo

## Objective

Review the provided training script and identify risks that could make the resulting entitlement score unreliable or difficult to audit.

---

## Findings

### 1. Target Leakage

The original script includes:

defaulted_in_next_12_months

as a training feature.

This appears to be information that becomes available after the scoring decision date.

If this feature is unavailable at scoring time, the model is learning future information.

This creates severe target leakage and may inflate validation performance.

Risk:
Validation metrics may appear strong while real-world performance is significantly worse.

---

### 2. Business Policy Mixed With Model Training

The script directly modifies the target:

df.loc[df["pm_kisan_status"] == "No",
"target_entitlement_score"] -= 150

This mixes business policy with model training.

Risks:

* Hard to audit
* Hard to explain
* Difficult to update policy independently
* Potential hidden bias

Business rules should be implemented separately from model training.

---

### 3. Unrealistic Validation Strategy

The original code uses:

train_test_split(... shuffle=True)

Random splitting allows future records to influence training.

For scoring systems, the objective is usually future prediction.

Risk:

Validation score may not represent future performance.

---

### 4. Label Encoding Risk

LabelEncoder was used on categorical variables.

Example:

Crop A → 0
Crop B → 1
Crop C → 2

The model may interpret this as an ordered relationship that does not exist.

OneHotEncoding is safer.

---

### 5. No Missing Value Strategy

The script does not define how missing values should be handled.

This may cause instability when scoring production data.

---

### 6. No Artifact Versioning

The original solution only saves:

xgboost_baseline.pkl

Missing:

* Feature list
* Validation summary
* Metadata
* Schema information

This reduces reproducibility.

---

### 7. No Feature Availability Review

Feature definitions were not provided.

The script assumes all features are available before scoring.

This assumption should be documented and validated.

---

## Changes Implemented

1. Removed potential leakage features.
2. Replaced random split with temporal validation.
3. Added preprocessing pipeline.
4. Added missing value handling.
5. Added reproducible artifact generation.
6. Separated business policy from model training.
7. Added metadata and validation summaries.

---

## Why Validation Is More Trustworthy

The revised approach validates on the most recent application year while training on earlier years.

This better simulates how the model would be used in production.

Limitations:

* Application year is only a proxy for true scoring time.
* Feature generation windows are unknown.

---

## Remaining Limitations

### One Thing I Do Not Trust Yet

I cannot fully verify feature availability because feature definitions and computation windows were not provided.

### One Thing I Would Improve With More Time

I would add feature-level explainability using SHAP and generate stable reason codes for score explanations.
