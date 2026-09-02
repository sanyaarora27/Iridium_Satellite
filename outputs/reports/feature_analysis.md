# Feature importance and channel-dominance analysis

## A. Random Forest feature importance (Task list Step 10)

Overall Random Forest accuracy on the 28 hand-crafted features:
**24.5%** against a chance level of 20.8%.

**Interpretation caveat.** Because overall accuracy is at or near chance,
these importances describe which features the forest *split on*, not which
features are genuinely informative. Impurity-based importance also favours
continuous, high-cardinality features irrespective of predictive value.
The ranking below should therefore be read as "where the model looked",
and interpreted together with Section B.

### Top 10 features

| Rank | Feature | Importance | R² explained by signal level | Likely source |
|-----:|---------|-----------:|-----------------------------:|---------------|
| 1 | `kurt_I` | 0.0504 | 0.318 | mixed channel + other |
| 2 | `spectral_centroid` | 0.0459 | 0.001 | not signal-level driven |
| 3 | `mean_I` | 0.0444 | 0.028 | not signal-level driven |
| 4 | `mean_Q` | 0.0443 | 0.025 | not signal-level driven |
| 5 | `kurt_Q` | 0.0443 | 0.339 | mixed channel + other |
| 6 | `skew_I` | 0.0437 | 0.008 | not signal-level driven |
| 7 | `papr` | 0.0436 | 0.448 | mixed channel + other |
| 8 | `median_I` | 0.0435 | 0.310 | mixed channel + other |
| 9 | `median_Q` | 0.0435 | 0.301 | mixed channel + other |
| 10 | `skew_Q` | 0.0429 | 0.006 | not signal-level driven |


## B. Channel-dominance analysis

Each feature was regressed against `level`, the receiver's estimated
signal strength. `level` depends on satellite range, elevation angle and
atmospheric path -- that is, on **geometry at the moment of capture**, not
on transmitter hardware.

- Features whose variance is **majority-explained** by signal level
  (R² > 0.5): **13 of 28**
- Most channel-dominated feature: `std_I` (R² = 0.991, Spearman ρ = 0.994)

A feature with high R² against `level` is largely a restatement of how
strong the received signal was. Two messages from the *same* satellite at
different elevations will differ more in these features than two messages
from *different* satellites at similar elevations -- which is precisely the
condition under which per-message classification fails.


## C. Metadata-only control experiment

A Random Forest was trained using **only** channel metadata
(level, noise, ra_alt, center_frequency) and no
signal-derived features at all.

| Model input | Accuracy |
|-------------|---------:|
| 28 hand-crafted signal features | 24.5% |
| Channel metadata only (no signal) | 25.2% |
| Chance (majority class) | 20.8% |

**The two are equivalent.** A model that never observes the
IQ waveform performs as well as one built on 28 signal-derived features.
The hand-crafted feature set therefore contributes no transmitter-specific
information beyond what channel geometry already supplies. Whatever small
margin exists above chance is attributable to capture geometry, not to
hardware fingerprints.


## D. Implications

1. **For the feature set.** Amplitude-scale features (signal power, min/max,
   FFT magnitude) are dominated by received signal strength. Any future
   feature set must be normalised per message so that scale is removed and
   only *shape* remains.

2. **For evaluation design.** Messages captured during the same satellite
   pass share channel conditions. A random train/test split can therefore
   let a model exploit pass-level rather than transmitter-level structure.
   Pass-aware (GroupKFold) evaluation is required for any positive result
   to be credible.

3. **For the security argument.** RF fingerprinting is proposed as a
   compensating control for Iridium's lack of cryptographic authentication.
   This analysis shows that a naive feature-based implementation measures
   propagation geometry rather than transmitter identity. As a control it
   would fail open under exactly the conditions an adversary can arrange --
   transmitting at a comparable range and elevation. Control effectiveness
   is contingent on feature robustness under channel variation, which makes
   detection maturity a risk-treatment input rather than a solved problem.
