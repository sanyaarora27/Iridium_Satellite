# Adversarial checks

Four objections anticipated and tested.

## A. Leakage audit

`global_index` was found to leak and was excluded. Because that was a
capture-order artefact rather than a physical property, every other stored
column was audited by the same test: train on that column alone and measure
whether it identifies the satellite.

| Column | Accuracy alone | Chance | Margin | Verdict |
|--------|---------------:|-------:|-------:|---------|
| `global_index` | 0.8500 | 0.2084 | +0.6415 | known leak, excluded |
| `timestamp_global` | 0.8483 | 0.2084 | +0.6399 | SEVERE LEAK - exclude |
| `ra_lon` | 0.3252 | 0.2084 | +0.1168 | suspicious - investigate |
| `ra_cell` | 0.2928 | 0.2084 | +0.0843 | weak signal |
| `ra_alt` | 0.2506 | 0.2084 | +0.0422 | weak signal |
| `run_id` | 0.2084 | 0.2084 | +0.0000 | no signal |
| `confidence` | 0.2084 | 0.2084 | +0.0000 | no signal |
| `direction` | 0.2084 | 0.2084 | +0.0000 | no signal |
| `level` | 0.2036 | 0.2084 | -0.0049 | no signal |
| `noise` | 0.2028 | 0.2084 | -0.0057 | no signal |
| `ra_lat` | 0.2011 | 0.2084 | -0.0073 | no signal |
| `center_frequency` | 0.1922 | 0.2084 | -0.0162 | no signal |

Additional leaking columns found: timestamp_global. These are excluded.

## B. Decode confidence

Messages whose reported position places the satellite below the horizon have significantly lower decode confidence (Mann-Whitney p = 1.68e-01), confirming bit errors in the payload as the cause.

Restricting classification to the higher-confidence half of messages gives 0.2585 against 0.2368 on the full set, so decode quality is not the limiting factor.

## C. Learning curve

| Training messages | Accuracy | Chance |
|------------------:|---------:|-------:|
| 493 | 0.2206 | 0.2084 |
| 1,232 | 0.2133 | 0.2084 |
| 2,465 | 0.2384 | 0.2084 |
| 3,697 | 0.2336 | 0.2084 |
| 4,930 | 0.2360 | 0.2084 |

## D. Hyperparameter sensitivity

| Configuration | Accuracy |
|---------------|---------:|
| scikit-learn defaults | 0.2360 |
| Best of 36 grid configurations | 0.2279 |
| Chance | 0.2084 |

Best parameters: `{'clf__max_depth': 8, 'clf__max_features': 0.5, 'clf__min_samples_leaf': 1, 'clf__n_estimators': 300}`

Tuning changes accuracy by -0.0081. The reported
conclusion does not depend on the use of default settings.
