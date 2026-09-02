# Spoofing attack in feature space (Step 13)

## Design

For an ordered pair of satellites (source S, target T), genuine messages
from S are moved toward T's class mean in feature space:

    x' = x + alpha * (mu_T - mu_S)

alpha = 0 leaves the message unmodified; alpha = 1 places it exactly at the
target's mean. The attack succeeds when the classifier assigns x' to T. The
reported quantity is the smallest alpha at which at least
50% of source messages are misclassified as the target.

Class means are estimated from the training split only, reflecting an
adversary who profiles a target from observed traffic.

## A. Attack on the 28 waveform features

Baseline accuracy: 24.5% (chance is approximately 20.8%).

Because this classifier does not perform above chance
(McNemar p = 0.12), impersonation succeeds
**18.9% of the time with no
modification whatsoever**. There is no meaningful attack to mount: the
classifier already assigns source messages to arbitrary classes at
approximately the base rate.

This is the correct result to report for a control that does not
discriminate. An attacker need not manipulate any waveform property,
because the defender cannot distinguish transmitters in the first place.

## B. Attack on the receiver-side metadata model

Baseline accuracy: 27.4%. This is the only model in
the project that classifies above chance (McNemar p = 0.0002), so it is the
only one for which the attack is meaningful.

| Source | Target | alpha required | Largest per-feature shift | Feature moved most |
|-------:|-------:|---------------:|--------------------------:|--------------------|

Median alpha required across impersonated pairs:
nan. Median largest per-feature shift:
nan within-class standard deviations.

Interpreting the shift in sigma units matters. A perturbation smaller than
one within-class standard deviation is smaller than the variation the
receiver already observes between genuine messages from the same
transmitter, and therefore cannot be flagged as anomalous without also
rejecting legitimate traffic.

## C. Which inputs are easiest to manipulate?

Each input was perturbed on its own, with all others left untouched, and
the resulting impersonation rate recorded. Numerical ease is reported
alongside how directly an adversary governs the quantity in practice.

| Model | Input | Impersonation rate | Attacker control | Basis |
|-------|-------|-------------------:|------------------|-------|
| 28 waveform features | `kurt_I` | 20.6% | indirect | derived from the waveform |
| 28 waveform features | `papr` | 19.6% | indirect | derived from the waveform |
| 28 waveform features | `min_I` | 19.4% | indirect | derived from the waveform |
| 28 waveform features | `kurt_Q` | 19.4% | indirect | derived from the waveform |
| 28 waveform features | `median_I` | 19.3% | indirect | derived from the waveform |
| 28 waveform features | `skew_Q` | 19.3% | indirect | derived from the waveform |
| 28 waveform features | `zero_crossing_rate` | 19.2% | indirect | derived from the waveform |
| 28 waveform features | `occupied_bandwidth` | 19.2% | indirect | derived from the waveform |
| 28 waveform features | `iqr_I` | 19.2% | indirect | derived from the waveform |
| 28 waveform features | `iq_correlation` | 19.2% | indirect | derived from the waveform |
| 28 waveform features | `max_I` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `skew_I` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `min_Q` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `peak_frequency` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `std_I` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `max_Q` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `mean_I` | 19.1% | indirect | derived from the waveform |
| 28 waveform features | `median_Q` | 19.0% | indirect | derived from the waveform |
| 28 waveform features | `signal_power` | 19.0% | indirect | derived from the waveform |
| 28 waveform features | `fft_mean_magnitude` | 19.0% | indirect | derived from the waveform |
| 28 waveform features | `mean_Q` | 19.0% | indirect | derived from the waveform |
| 28 waveform features | `iqr_Q` | 18.9% | indirect | derived from the waveform |
| 28 waveform features | `bandwidth` | 18.9% | indirect | derived from the waveform |
| 28 waveform features | `iq_ratio` | 18.9% | indirect | derived from the waveform |
| 28 waveform features | `std_Q` | 18.9% | indirect | derived from the waveform |
| 28 waveform features | `var_Q` | 18.8% | indirect | derived from the waveform |
| 28 waveform features | `var_I` | 18.8% | indirect | derived from the waveform |
| 28 waveform features | `spectral_centroid` | 18.8% | indirect | derived from the waveform |
| receiver-side metadata | `level` | 20.0% | indirect | follows from transmit power and range |
| receiver-side metadata | `noise` | 19.2% | none | receiver noise floor, not attacker-set |
| receiver-side metadata | `center_frequency` | 16.7% | direct | attacker chooses transmit frequency |

## D. Security interpretation

The two models fail in different ways, and the distinction matters for the
control assessment.

The waveform-feature model offers no protection because it does not
discriminate: impersonation succeeds without any manipulation. A control
that cannot separate legitimate transmitters cannot detect an illegitimate
one.

The receiver-side metadata model does discriminate, but what it
discriminates on is operating context -- carrier frequency, received power
and noise floor -- rather than transmitter hardware. Carrier frequency is
set directly by the transmitter. Received power follows from transmit power
and range, both of which an adversary chooses. Consequently an attacker who
transmits on the appropriate simplex channel at a plausible power from a
plausible location satisfies the model's expectations without imitating any
physical characteristic of the genuine satellite.

Neither model, therefore, provides transmitter authentication. The first
lacks discriminative power; the second discriminates on quantities under
the adversary's control.
