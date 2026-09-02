# Results table (Task list Step 9)

**Task:** classify which Iridium satellite transmitted a given message.

- Satellites: 5
- Samples: 6,163 (4,930 train / 1,233 test)
- Split: stratified 80/20, seed 42
- Preprocessing: StandardScaler fitted inside a Pipeline on training folds only
- Hyperparameters: scikit-learn defaults, no tuning

| Model | Features used | Accuracy | Macro F1 | Notes |
|-------|---------------|---------:|---------:|-------|
| Dummy (chance) | none (majority class) | 0.2084 | 0.0690 | Chance reference; predicts majority class only |
| Logistic Regression | 28 hand-crafted | 0.2311 | 0.2232 | Not distinguishable from chance (CI overlaps reference) |
| Decision Tree | 28 hand-crafted | 0.2157 | 0.2156 | Not distinguishable from chance (CI overlaps reference) |
| Random Forest | 28 hand-crafted | 0.2360 | 0.2326 | Not distinguishable from chance (CI overlaps reference) |
| SVM (RBF) | 28 hand-crafted | 0.2238 | 0.2179 | Not distinguishable from chance (CI overlaps reference) |
| k-NN (k=5) | 28 hand-crafted | 0.2141 | 0.2121 | Not distinguishable from chance (CI overlaps reference) |
| Naive Bayes | 28 hand-crafted | 0.2182 | 0.1886 | Not distinguishable from chance (CI overlaps reference) |
| Neural Net (MLP) | 28 hand-crafted | 0.2328 | 0.2308 | Not distinguishable from chance (CI overlaps reference) |
| Random Forest | 4 channel metadata (level, noise, ra_alt, center_frequency) | 0.2506 | 0.2481 | Control: no waveform data used at all |

## Reading this table

The `Features used` column carries the argument. The first group of rows all
consume the same 28 hand-crafted signal features and none of them separates
from the chance reference. The final row uses **no waveform data whatsoever**
-- only channel metadata describing where the satellite was and how strong
the signal arrived -- and performs comparably.

The conclusion is therefore not simply that the models failed. It is that the
28 hand-crafted features supply no transmitter-specific information beyond
what capture geometry already provides. Whatever small margin exists above
chance is attributable to orbital geometry rather than to hardware
fingerprints.

Per the task brief, the aim at this stage is not to match SatIQ's reported
performance but to build a baseline that is fully understood and explainable.
This table documents both the result and the reason for it.
