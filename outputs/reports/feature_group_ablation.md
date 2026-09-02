# Feature-group ablation

## Method

All experiments used the same stratified 80/20 split (random seed 42) and a 200-tree Random Forest.
Median imputation was fitted only on the training partition. `global_index`, `satellite_id`, and chronology-derived metadata were not predictive inputs.
The top-10 subset was selected using Random Forest importance computed on the training partition only.

No explicit raw-phase feature group exists in the primary v1 28-feature representation; phase/CFO-oriented descriptors belong to later feature experiments.

## Results

| Experiment | n | Accuracy | Macro F1 | P_D | FRR / P_FA | FAR | EER | Δ Acc. pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All 28 v1 features | 28 | 0.245 | 0.242 | 0.244 | 0.756 | 0.189 | 0.471 | +0.00 |
| Without time-domain statistics | 10 | 0.225 | 0.224 | 0.224 | 0.776 | 0.194 | 0.477 | -2.03 |
| Without amplitude/power | 16 | 0.219 | 0.216 | 0.218 | 0.782 | 0.195 | 0.477 | -2.60 |
| Without frequency-domain | 23 | 0.236 | 0.232 | 0.235 | 0.765 | 0.191 | 0.475 | -0.89 |
| Without I/Q relationship + temporal | 25 | 0.242 | 0.238 | 0.240 | 0.760 | 0.190 | 0.474 | -0.32 |
| Top-10 RF features only | 10 | 0.207 | 0.202 | 0.205 | 0.795 | 0.199 | 0.484 | -3.81 |

## Training-only top 10 features

1. `kurt_I` — importance 0.05037
2. `spectral_centroid` — importance 0.04589
3. `mean_I` — importance 0.04440
4. `mean_Q` — importance 0.04435
5. `kurt_Q` — importance 0.04430
6. `skew_I` — importance 0.04374
7. `papr` — importance 0.04355
8. `median_I` — importance 0.04351
9. `median_Q` — importance 0.04349
10. `skew_Q` — importance 0.04287

## Interpretation rule

A reduction after removing a feature group indicates that the group contributed useful predictive information. It does not, by itself, establish that the information is transmitter-hardware specific. The result must be interpreted alongside the existing channel-dominance and leakage analyses.