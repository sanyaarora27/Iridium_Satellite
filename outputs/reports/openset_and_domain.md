# Feature sets, cross-domain generalisation, and open-set rejection

Three evaluations drawn from the 2026 satellite-security literature.

## Evaluation provenance

- Pass definition: timestamp-gap inferred pass, with a 300-second
  gap threshold, grouped per satellite from timestamp_global.
- Feature set used for cross-domain and open-set evaluation: all combined;
  pre-specified all combined representation; not selected from this test set.
- Preprocessing: median imputation and StandardScaler are fitted inside each
  training pipeline and applied unchanged to evaluation data.
- Open-set threshold: none is fitted; AUROC is threshold-free and uses maximum
  class probability only as a ranking score (not applicable; AUROC is threshold-free).
- Random seed: 42.

## A. Feature set comparison

A March 2026 theoretical study of satellite RF fingerprint limits reports
that IQ imbalance may carry insufficient identifying information under some
modulation schemes, while power-amplifier nonlinearities are more reliable.
The v2 set tested the former and did not improve on v1; v3 adds AM/PM
conversion, envelope compression and spectral-regrowth features to test the
latter.

| Feature set | Features | Accuracy | 95% CI | p vs chance |
|-------------|---------:|---------:|:------:|------------:|
| chance | 0 | 0.2084 | [0.187, 0.232] | 1.0000 |
| v1 (hand-crafted) | 28 | 0.2449 | [0.222, 0.269] | 0.0369 |
| v2 (amplitude-invariant) | 26 | 0.2352 | [0.211, 0.259] | 0.1223 |
| v3 (PA nonlinearity) | 20 | 0.2206 | [0.198, 0.244] | 0.4941 |
| all combined | 74 | 0.2490 | [0.225, 0.274] | 0.0208 |

## B. Cross-domain generalisation

Channel-robustness research reports accuracy within a channel condition and
across conditions separately, treating the gap as the result. All previous
evaluations in this project were within-domain: a random split places every
beam and every pass on both sides. Beam fixes the transmit antenna pattern;
pass fixes geometry and time.

| Domain | Source domains | Target domains | Train satellites | Test satellites | Train n | Test n | Within margin | Cross margin | Loss |
|--------|---------------:|---------------:|------------------|-----------------|---------:|--------:|--------------:|-------------:|-----:|
| beam | 24 | 25 | [51, 85, 87, 92, 109] | [51, 85, 87, 92, 109] | 2932 | 3231 | +0.0321 | +0.0269 | +0.0052 |
| pass_id | 26 | 27 | [51, 85, 87, 92, 109] | [51, 85, 87, 92, 109] | 2991 | 3172 | +0.0321 | +0.0211 | +0.0110 |

## C. Open-set rejection

Operational monitoring requires detecting unknown transmitters rather than
assigning every signal to a fixed set, and current guidance lists a
rejection mechanism as a required component of physical-layer
authentication. Every model built in this project is closed-set.

Each satellite is held out in turn, a model trained on the remaining four,
and maximum class probability used as a confidence score.

| Held out | Known train satellites | Unknown test satellites | Known test n | Unknown test n | AUROC | Mean confidence (known) | Mean confidence (unknown) |
|---------:|------------------------|--------------------------|--------------:|----------------:|------:|------------------------:|--------------------------:|
| 51 | [85, 87, 92, 109] | [51] | 995 | 1,188 | 0.5205 | 0.345 | 0.341 |
| 85 | [51, 87, 92, 109] | [85] | 978 | 1,277 | 0.5070 | 0.328 | 0.327 |
| 87 | [51, 85, 92, 109] | [87] | 987 | 1,232 | 0.5330 | 0.347 | 0.339 |
| 92 | [51, 85, 87, 109] | [92] | 977 | 1,282 | 0.5048 | 0.345 | 0.344 |
| 109 | [51, 85, 87, 92] | [109] | 996 | 1,184 | 0.5161 | 0.338 | 0.334 |

Mean AUROC 0.5163, where 0.5 indicates no ability to
distinguish an unenrolled transmitter from an enrolled one.
