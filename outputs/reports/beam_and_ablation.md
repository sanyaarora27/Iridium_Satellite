# Metadata ablation and within-beam analysis

## A. Which metadata column carries the signal?

McNemar's test showed that a Random Forest using four channel metadata
values classifies above chance, while the same model using 28 hand-crafted
signal features does not. The variance decomposition then showed that
`level` explains almost none of the variation associated with satellite
identity, so `level` cannot be responsible. This ablation isolates the
column that is.

| Variant | Accuracy | Chance | p (McNemar) |
|---------|---------:|-------:|------------:|
| all 4 columns | 0.2644 | 0.2084 | 0.0010 |
| only level | 0.2036 | 0.2084 | 0.7986 |
| only noise | 0.2028 | 0.2084 | 0.7684 |
| only ra_alt | 0.2482 | 0.2084 | 0.0243 |
| only center_frequency | 0.1930 | 0.2084 | 0.3792 |
| without level | 0.2636 | 0.2084 | 0.0014 |
| without noise | 0.2295 | 0.2084 | 0.2225 |
| without ra_alt | 0.2733 | 0.2084 | 0.0002 |
| without center_frequency | 0.1979 | 0.2084 | 0.5450 |
| receiver-side only | 0.2733 | 0.2084 | 0.0002 |
| payload only | 0.2482 | 0.2084 | 0.0243 |


## B. Within-beam classification

Each Iridium satellite transmits through 48 spot beams with distinct
antenna patterns. Holding the beam index constant fixes the approximate
angle from boresight, and therefore the transmit antenna gain toward the
receiver -- a tighter channel control than fixing elevation, which
constrains only the propagation path.

Beams with at least 250 messages were analysed separately.
Because several beams are tested, per-beam p-values are corrected using the
Holm-Bonferroni step-down procedure; without correction, testing twenty
beams at alpha = 0.05 would be expected to yield one spurious result.

| Beam | Messages | Accuracy | 95% CI | Chance | Margin | p | Significant (Holm) |
|---|---:|---:|:---:|---:|---:|---:|---|
| 9 | 275 | 0.4364 | [0.309, 0.582] | 0.2909 | +0.1455 | 0.1516 | no |
| 1 | 384 | 0.2857 | [0.182, 0.390] | 0.2727 | +0.0130 | 1.0000 | no |

Mean margin over chance across beams: +0.0792. 0 of 2 beams remain significant after correction.
