# Statistical rigour checks

Five corrections to the earlier analysis.

## A. Pass-aware evaluation

Messages captured during the same timestamp-gap inferred pass share channel
conditions. This is an operational grouping heuristic, not a physical or
orbital pass proven from ephemeris data. A random train/test split can place
messages from one inferred pass on both sides of the split, allowing a model
to score above chance by recognising the pass rather than the transmitter.
Passes were recovered from timestamp_global gaps (53 passes
identified) using the canonical 300-second threshold, converted
from nanoseconds to seconds, grouped per satellite, and tie-broken by
global_index.

| Split | Accuracy | Chance |
|-------|---------:|-------:|
| Random (StratifiedKFold) | 0.2338 ± 0.0112 | 0.2080 |
| Pass-aware (StratifiedGroupKFold) | 0.1570 ± 0.0375 | 0.0139 |

Difference: +0.0768.

Bootstrap intervals in this report resample messages and describe
message-level uncertainty; they must not be interpreted as independent
inferred-pass uncertainty. McNemar results below are based on a message-level
split and support message-level comparisons, not pass-level generalisation
claims. The Mahalanobis analysis uses the full dataset and is descriptive,
not held-out predictive validation.

## B. Paired classifier comparison

Accuracy point estimates were previously used to describe one model as
performing better than another. Because all models are evaluated on the
same test set their errors are paired, and the correct test is McNemar's,
which considers only the discordant predictions.

| Model | Accuracy | 95% CI (bootstrap) |
|-------|---------:|:------------------:|
| 28 hand-crafted features | 0.2360 | [0.2133, 0.2603] |
| Dummy (chance) | 0.2084 | [0.1873, 0.2320] |
| 4 channel metadata | 0.2506 | [0.2271, 0.2749] |

McNemar, 28 features vs chance: p = 0.1197
(not significant at alpha = 0.05).

McNemar, channel metadata vs chance: p = 0.0099
(significant).

McNemar, metadata vs 28 features: p = 0.4188
(not significant).

## C. Mahalanobis separability

The earlier separability figure used Euclidean distance between class means
with an ad-hoc normalisation and an invented threshold. Euclidean distance
is unsuitable here because the feature set contains near-duplicates
(`std_I`, `var_I` and `iqr_I` all measure spread), so correlated quantities
are counted repeatedly. Mahalanobis distance uses the pooled within-class
covariance and is the standard measure.

Best pairwise Mahalanobis distance: **D = 0.112**

D is expressed in pooled within-class standard deviations. D below 1
indicates that class means are closer together than the typical scatter
within a class; D above 3 would indicate well-separated classes.

## D. What drives `level`

The earlier analysis reported that many features correlate with `level` at
R² above 0.98 and described this as channel dominance. That inference was
partly circular: `level` is itself an estimate of received amplitude, so any
amplitude-derived feature must correlate with it. The substantive question
is what drives `level`.

| Predictor | Type | R² |
|-----------|------|---:|
| receiver noise floor | continuous | 0.120 |
| beam (ra_cell) | categorical | 0.089 |
| elevation angle | continuous | 0.039 |
| satellite identity | categorical | 0.001 |

The defensible claim is therefore that the hand-crafted features are
dominated by **signal amplitude**. The extent to which that amplitude is
attributable to orbital geometry is given by the table above and should be
stated as measured rather than assumed.

## E. Within-beam classification

Iridium satellites transmit through multiple spot beams with different
antenna patterns. Holding the beam constant is a tighter channel control
than holding elevation constant, because it fixes the antenna gain toward
the receiver as well as the approximate geometry.

| Beam | Messages | Accuracy | Chance |
|---|---:|---:|---:|
| 1 | 384 | 0.3358 | 0.2630 |
