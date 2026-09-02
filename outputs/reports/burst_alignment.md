# Burst alignment diagnostic

## Why this check was performed

All 28 features are statistics computed over an 11,000-sample
(440 us) capture window. If the transmitted
burst occupies only part of that window, every feature is a blend of signal
and receiver noise, with the blend ratio set by demodulator trigger timing
rather than by the transmitter. Two messages from the same satellite would
then differ in every amplitude feature for reasons unrelated to hardware.

## Burst geometry

Measured on 750 messages across 5 satellites.

| Quantity | Value |
|----------|------:|
| Window length | 11,000 samples (440 us) |
| Burst fraction (median) | 1.000 |
| Burst fraction (std across messages) | 0.003 |
| Burst fraction (min / max) | 0.958 / 1.000 |
| Burst start (median sample) | 1 |
| Burst start (std) | 26 samples |

**Burst fills essentially the whole window. Window misalignment is NOT a confound.**

## Does trimming to the burst improve class separation?

One-way ANOVA F-statistic per feature, computed on the full window and on
the burst region only. F is the ratio of between-class to within-class
variance; F = 1 indicates no class separation.

| Feature | F (full window) | F (burst only) | Change |
|---------|----------------:|---------------:|-------:|
| `std_I` | 2.08 | 2.08 | +0% |
| `std_Q` | 2.07 | 2.07 | +0% |
| `signal_power` | 2.70 | 2.70 | +0% |
| `papr` | 1.77 | 1.76 | -1% |
| `kurt_I` | 2.56 | 2.49 | -3% |
| `iq_corr` | 2.25 | 2.25 | -0% |
| **mean** | **2.24** | **2.22** | **-1%** |

**Trimming does NOT improve discriminative power. Window alignment is not what is limiting the classifier; the features themselves carry no transmitter information.**

## Interpretation

This diagnostic distinguishes two competing explanations for the
chance-level baseline:

1. *The features are fine, but the extraction window is misaligned*, so
   signal is diluted by noise in a message-dependent way.
2. *The features themselves carry no transmitter-specific information.*

The F-statistic comparison above discriminates between them directly: if
explanation 1 held, restricting the computation to the burst would raise
the F-statistics materially. The measured result is reported above.
