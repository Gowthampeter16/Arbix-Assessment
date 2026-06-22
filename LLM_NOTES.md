# LLM_NOTES

## Tools Used

* ChatGPT
* Official Scikit-Learn Documentation
* XGBoost Documentation

---

## Where They Were Used

* Reviewing leakage risks
* Designing validation strategy
* Reviewing preprocessing choices
* Drafting documentation

---

## Three Actual Prompts Used

Prompt 1:

Review this ML training script and identify data leakage, validation risks and reproducibility issues.

Prompt 2:

Suggest a safer validation strategy for a scoring model where future performance matters more than leaderboard accuracy.

Prompt 3:

Generate an audit memo explaining why target leakage and random train-test splits can inflate validation scores.

---

## Suggestions Accepted

* Remove defaulted_in_next_12_months from training features.
* Use temporal validation instead of random splitting.
* Use sklearn Pipeline for preprocessing and model training.
* Save metadata and validation artifacts.

---

## Suggestions Rejected Or Corrected

Suggestion:

Perform aggressive hyperparameter tuning.

Reason Rejected:

The assessment prioritizes safety, explainability and validation rather than leaderboard performance.

---

## Personally Verified

Leakage reasoning
Validation split logic
Feature preprocessing boundary
Saved artifacts
Training execution
Generated outputs
Model serialization
