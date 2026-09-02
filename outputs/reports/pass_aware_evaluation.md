# Pass-aware classifier evaluation

## Purpose

This experiment evaluates whether the classifiers can identify a satellite from
messages belonging to previously unseen satellite passes. All messages assigned
to one pass remain entirely within either the training fold or the test fold.

## Experimental setup

- **Messages:** 6,163
- **Satellites:** 5
- **RF features:** 28
- **Inferred passes:** 53
- **Pass definition:** timestamp-gap inferred pass, grouped per satellite; a new inferred pass begins after a time gap greater than 300 seconds
- **Timestamp source:** timestamp_global, converted from nanoseconds to seconds
- **Grouping scope:** per satellite; global_index is the deterministic tie-breaker
- **Evaluation:** 5-fold StratifiedGroupKFold grouped cross-validation
- **Interpretation:** this is an operational grouping heuristic, not a physical or orbital pass proven from ephemeris data
- **Leakage control:** no pass appears in both training and test data within a fold

## Results

| Model | Mean accuracy | Accuracy SD | Mean macro-F1 | Minimum fold | Maximum fold |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.1900 | 0.0117 | 0.1699 | 0.1808 | 0.2110 |
| Neural Net (MLP) | 0.1698 | 0.0349 | 0.1585 | 0.1345 | 0.2293 |
| k-NN (k=5) | 0.1675 | 0.0245 | 0.1535 | 0.1275 | 0.1976 |
| Random Forest | 0.1555 | 0.0413 | 0.1527 | 0.1151 | 0.2329 |
| Gaussian Naive Bayes | 0.1310 | 0.0757 | 0.0755 | 0.0412 | 0.2439 |
| Logistic Regression | 0.1003 | 0.0423 | 0.1084 | 0.0624 | 0.1817 |
| SVM (RBF) | 0.0683 | 0.0467 | 0.0763 | 0.0323 | 0.1585 |
| Baseline (most frequent) | 0.0139 | 0.0178 | 0.0054 | 0.0006 | 0.0455 |


## Interpretation

The strongest pass-aware model was **Decision Tree**, with mean accuracy
**19.0%** and mean macro-F1
**0.170**.

Pass-aware performance should be compared with the random message-level split.
A substantial reduction indicates that the random split benefited from
pass-specific channel conditions rather than a stable transmitter fingerprint.
