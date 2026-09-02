# v1 vs v2 feature comparison

## Design

The v2 feature set was constructed to address two diagnosed weaknesses in
v1: amplitude dominance (13 of 28 features had R² above 0.98 against the
receiver's amplitude estimate) and loss of temporal structure (all v1
features were aggregates, unchanged by reordering the samples).

All models use the same train/test split, the same Random Forest settings,
and the same chance baseline, so differences are attributable to the
features alone.

| Model | Features | Accuracy | 95% CI | Macro F1 | p vs chance |
|-------|---------:|---------:|:------:|---------:|------------:|
| chance (majority class) | 0 | 0.2084 | [0.187, 0.232] | 0.0690 | - |
| v1 (hand-crafted) | 28 | 0.2490 | [0.226, 0.273] | 0.2451 | 0.0208 |
| v2 (amplitude-invariant) | 26 | 0.2425 | [0.219, 0.267] | 0.2384 | 0.0484 |
| v1 + v2 combined | 54 | 0.2320 | [0.209, 0.255] | 0.2258 | 0.1753 |

Direct comparison of v2 against v1 (McNemar): p = 0.7122.

## v2 feature importance

| Rank | Feature | Importance |
|-----:|---------|-----------:|
| 1 | `envelope_kurt` | 0.0414 |
| 2 | `cfo_drift_hz_per_ms` | 0.0405 |
| 3 | `cfo_hz` | 0.0401 |
| 4 | `seg2_power_ratio` | 0.0399 |
| 5 | `iq_gain_imbalance_db` | 0.0398 |
| 6 | `seg1_power_ratio` | 0.0398 |
| 7 | `seg3_power_ratio` | 0.0395 |
| 8 | `skew_i_norm` | 0.0394 |
| 9 | `cfo_skew` | 0.0392 |
| 10 | `evm_proxy` | 0.0392 |

## Interpreting a CFO-driven result

Carrier frequency offset combines two sources. Transmitter oscillator error
is a hardware property, typically of order 1 kHz. Doppler shift is a
geometric property, reaching approximately +/- 40 kHz at Iridium's L-band
for a satellite in low Earth orbit. Doppler is therefore expected to
dominate the measured offset.

If `cfo_hz` ranks highly and is strongly associated with elevation angle,
the model is keying on orbital geometry rather than on oscillator
characteristics. That has the same authentication weakness already
identified for the receiver-side metadata model: an adversary transmitting
from a plausible position produces a plausible offset, without imitating
any hardware property.

Separating the two would require subtracting predicted Doppler computed
from satellite ephemeris. In this subset 48.6% of reported positions failed
a physical plausibility check, so that correction cannot be applied
reliably here and is left as future work.
