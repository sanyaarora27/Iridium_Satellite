# Feature list and physical interpretation

The 28 hand-crafted features extracted per message by
`03_extract_features.py`, each with a plain-English description and the
physical effect it primarily reflects. The final column records what the
channel-dominance analysis (script 05) found: how much of the feature's
variation is explained by received signal amplitude (`level`).

Physical-effect categories follow the supervisor's Step 10 request:
**hardware** (transmitter imperfection), **power** (received signal
strength), **channel** (propagation and geometry), **frequency**
(oscillator or Doppler), **structural** (a property of the modulation or
protocol rather than the transmitter).

---

## Time-domain amplitude and shape

| Feature | Meaning | Primary effect | R² vs level |
|---------|---------|----------------|:-----------:|
| `mean_I` | Average of the in-phase component. A non-zero value indicates a DC offset introduced by the modulator. | hardware (small) / structural | low |
| `mean_Q` | Average of the quadrature component; DC offset on the Q channel. | hardware (small) / structural | low |
| `var_I` | Variance of I; the spread of in-phase amplitude, which scales with received power. | power | high |
| `var_Q` | Variance of Q; scales with received power. | power | high |
| `std_I` | Standard deviation of I; the square root of the variance, again a power measure. | power | **0.99** |
| `std_Q` | Standard deviation of Q. | power | **0.99** |
| `max_I` | Largest in-phase sample; a peak-amplitude measure sensitive to received power. | power | high |
| `max_Q` | Largest quadrature sample. | power | high |
| `min_I` | Smallest (most negative) in-phase sample. | power | high |
| `min_Q` | Smallest quadrature sample. | power | high |
| `median_I` | Median of I; a robust centre estimate, again reflecting DC offset. | hardware (small) | low |
| `median_Q` | Median of Q. | hardware (small) | low |
| `iqr_I` | Interquartile range of I; a robust spread measure, scaling with power. | power | **0.99** |
| `iqr_Q` | Interquartile range of Q. | power | **0.99** |

## Distribution shape (scale-independent)

| Feature | Meaning | Primary effect | R² vs level |
|---------|---------|----------------|:-----------:|
| `skew_I` | Skewness of I; asymmetry of the amplitude distribution. Dimensionless, so independent of power. | hardware / structural | low |
| `skew_Q` | Skewness of Q. | hardware / structural | low |
| `kurt_I` | Kurtosis of I; tailedness of the distribution, sensitive to clipping or compression. | hardware | low |
| `kurt_Q` | Kurtosis of Q. | hardware | low |

## Power and I/Q relationship

| Feature | Meaning | Primary effect | R² vs level |
|---------|---------|----------------|:-----------:|
| `signal_power` | Mean of I²+Q²; the total received power. The most direct power measure. | power | **0.99** |
| `iq_ratio` | Ratio of I standard deviation to Q standard deviation; a crude measure of gain imbalance between the two channels. | hardware | low |
| `iq_correlation` | Correlation between I and Q; for ideal QPSK this is near zero, and a non-zero value indicates a quadrature phase error. | hardware | low |
| `papr` | Peak-to-average power ratio in dB; how far the largest peak exceeds the mean, sensitive to amplifier compression. | hardware / structural | low |
| `zero_crossing_rate` | Rate at which the in-phase component changes sign; a coarse frequency proxy. | frequency / structural | low |

## Frequency-domain

| Feature | Meaning | Primary effect | R² vs level |
|---------|---------|----------------|:-----------:|
| `fft_mean_magnitude` | Mean magnitude of the spectrum; an overall energy measure that scales with power. | power | **0.98** |
| `peak_frequency` | Frequency of the strongest spectral component; reflects carrier offset from oscillator error and Doppler. | frequency | low |
| `spectral_centroid` | Amplitude-weighted mean frequency; the spectral "centre of mass". | frequency / structural | low |
| `bandwidth` | Spread of the spectrum about its centroid; a measure of occupied bandwidth. | structural | low |
| `occupied_bandwidth` | Width of the band holding 90% of spectral energy. | structural | low |

---

## Summary

Of the 28 features, **13 have R² above 0.98 against received amplitude**
(the `var`, `std`, `iqr`, `max`, `min`, `signal_power` and
`fft_mean_magnitude` group). These measure how strong the signal arrived,
not which transmitter sent it, and received strength is set by range and
geometry rather than by hardware.

The features that could in principle carry hardware information — DC offset
(`mean`, `median`), gain and phase imbalance (`iq_ratio`,
`iq_correlation`), distribution shape (`skew`, `kurt`), and carrier offset
(`peak_frequency`) — are dimensionless or frequency-based and are not
dominated by amplitude. The classification results show that whatever
hardware signal these carry is too weak to separate the five satellites
reliably, which motivated the amplitude-invariant v2 features and the
power-amplifier-nonlinearity v3 features tested later.

`global_index` and `timestamp_global` appear in the raw data but are **not
features**: they encode capture order and were excluded from every model
after the leakage audit (script 15) found each classifies at ~85% by
encoding when a message was recorded rather than who sent it.
