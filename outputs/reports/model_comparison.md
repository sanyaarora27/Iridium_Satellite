# Baseline classifier comparison

## Experimental setup

- **Task:** multi-class classification of Iridium satellite transmitter ID
- **Classes:** 5 satellites (51, 85, 87, 92, 109)
- **Features:** 28 hand-crafted RF features per message
- **Split:** stratified 80% train / 20% test
- **Training samples:** 4,930
- **Test samples:** 1,233
- **Chance level (majority class):** 20.8%
- **Preprocessing:** StandardScaler fitted inside a Pipeline on training folds only
- **Hyperparameters:** scikit-learn defaults (no tuning at this stage)

## Results

| Model | Accuracy | 95% CI | Macro F1 | Notes |
|-------|---------:|:------:|---------:|-------|
| Dummy (chance) | 0.2084 | [0.187, 0.232] | 0.0690 | chance reference |
| Logistic Regression | 0.2311 | [0.208, 0.254] | 0.2232 |  |
| Decision Tree | 0.2157 | [0.194, 0.239] | 0.2156 |  |
| Random Forest | 0.2360 | [0.213, 0.260] | 0.2326 |  |
| SVM (RBF) | 0.2238 | [0.203, 0.247] | 0.2179 |  |
| k-NN (k=5) | 0.2141 | [0.191, 0.237] | 0.2121 |  |
| Naive Bayes | 0.2182 | [0.195, 0.242] | 0.1886 |  |
| Neural Net (MLP) | 0.2328 | [0.209, 0.256] | 0.2308 |  |


## Interpretation

The strongest model is **Random Forest** at 23.6% accuracy
(95% CI [21.3%, 26.0%]),
compared with a chance level of 20.8%.

**No model's confidence interval separates from the chance
reference.** The hand-crafted feature set, as extracted, does not carry
usable class-discriminative information for this task.

This is a genuine empirical finding rather than an implementation error, and
it is diagnosed further in Section X (channel-dominance analysis), which
quantifies how much of each feature's variance is explained by received
signal strength rather than by transmitter identity. Overlapping confidence
intervals also mean that ranking the models against one another here would
not be statistically supportable.

## Methodological notes

- Confidence intervals are computed by bootstrap resampling of the test set
  (1,000 resamples, percentile method). Where intervals overlap, differences
  between models are not statistically distinguishable.
- The train/test split is random over messages. Messages captured during the
  same satellite pass share channel conditions, so a random split may allow a
  model to exploit pass-level rather than transmitter-level information. A
  pass-aware (GroupKFold) evaluation is reported separately.
