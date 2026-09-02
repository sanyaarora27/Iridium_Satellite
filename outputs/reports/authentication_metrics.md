# Authentication error rates

## Why accuracy is the wrong metric

The supervisor's brief frames the security question as verification:

> If a signal claims to be satellite 1 but the RF fingerprint classifier
> does not classify it as satellite 1, then there may be spoofing.

That decision has two failure modes, and a control must be judged against
both. The False Reject Rate is the proportion of genuine traffic from a
satellite that fails to be recognised as that satellite, and would
therefore be flagged as spoofed. The False Accept Rate is the proportion of
messages from other transmitters that would be accepted under a claimed
identity.

Accuracy conceals this trade-off. A model at 27% accuracy rejects roughly
three quarters of authentic traffic, which no operator would deploy
irrespective of its resistance to attack.

## A. Argmax decision rule

| Model | Accuracy | Mean FRR | Mean FAR |
|-------|---------:|---------:|---------:|
| 28 waveform features | 24.5% | 75.6% | 18.9% |
| receiver-side metadata | 27.4% | 72.6% | 18.2% |

Per-satellite detail:

| Model | Satellite | Genuine messages | FRR | FAR |
|-------|----------:|-----------------:|----:|----:|
| 28 waveform features | 51 | 238 | 81.9% | 14.8% |
| 28 waveform features | 85 | 255 | 66.3% | 24.8% |
| 28 waveform features | 87 | 246 | 77.2% | 17.8% |
| 28 waveform features | 92 | 257 | 77.8% | 20.2% |
| 28 waveform features | 109 | 237 | 74.7% | 16.9% |
| receiver-side metadata | 51 | 238 | 73.1% | 18.8% |
| receiver-side metadata | 85 | 255 | 73.7% | 18.2% |
| receiver-side metadata | 87 | 246 | 72.8% | 19.0% |
| receiver-side metadata | 92 | 257 | 70.0% | 19.9% |
| receiver-side metadata | 109 | 237 | 73.4% | 14.9% |

## B. Equal Error Rate

Random Forests emit class probabilities, so the verifier can accept a
claimed identity only when its probability exceeds a threshold tau.
Raising tau tightens acceptance: FAR falls while FRR rises. The Equal Error
Rate is the crossing point, and is the standard summary used in the
biometric and RF-fingerprinting literature.

| Model | EER | Threshold |
|-------|----:|----------:|
| 28 waveform features | 47.1% | 0.20 |
| receiver-side metadata | 42.8% | 0.18 |
| SatIQ (Smailes et al., 2025) | 7.2% | - |

The best classical model reaches an EER of 42.8% against
SatIQ's published 7.2% on the same constellation --
approximately 6 times worse.

## C. Control assessment

At these error rates the control is unusable in both directions. Operating
at the equal error point would reject a large share of genuine satellite
traffic while admitting a comparable share of impersonations. Moving the
threshold to make either rate acceptable makes the other worse, and no
setting yields a workable operating point.

The gap to SatIQ is the substantive finding. Both approaches consume the
same physical-layer data from the same constellation. The difference lies
entirely in representation: hand-crafted summary statistics discard the
structure that a learned embedding preserves. Deep learning is not applied
here for its own sake but because the discriminative information is not
recoverable by simple aggregate statistics over the waveform.
