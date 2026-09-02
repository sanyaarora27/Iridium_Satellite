"""
15_extract_features_v2.py
=========================

PURPOSE
-------
A second feature set designed to address the two structural weaknesses
diagnosed in the v1 baseline.

  WEAKNESS 1 - AMPLITUDE DOMINANCE
    Thirteen of the twenty-eight v1 features have R^2 above 0.98 against
    the receiver's amplitude estimate, while satellite identity explains
    only 0.1% of that amplitude. Nearly half the feature set measured link
    budget rather than transmitter identity.

    Fix: every feature below is amplitude-invariant. Each message is scaled
    to unit average power before extraction, and features are constructed
    from ratios, angles or normalised quantities so that multiplying the
    whole waveform by a constant leaves them unchanged.

  WEAKNESS 2 - LOSS OF TEMPORAL STRUCTURE
    Every v1 feature is an aggregate over the full 11,000-sample window.
    The mean of I is unchanged if the samples are randomly shuffled, and
    the same is true of its variance, skewness and FFT magnitude. Hardware
    fingerprints, however, arise from ordering: oscillator drift across the
    burst, phase error through symbol transitions, amplifier settling.

    Fix: features below are computed from the instantaneous phase and
    frequency trajectory, from the envelope trend, and from consecutive
    sub-windows -- all of which change if the sequence is reordered.

REMOVING THE QPSK MODULATION
----------------------------
Instantaneous frequency cannot be read directly from an Iridium burst: the
QPSK data modulation imposes phase steps of +/- pi/2 and pi at every symbol
transition, and these dominate any phase derivative.

The standard remedy is the M-th power method (Viterbi & Viterbi, 1983).
For QPSK, raising the complex signal to the fourth power maps all four
constellation points onto a single point, cancelling the data modulation
and leaving four times the residual carrier:

    z(t)  = A * exp( j * (2*pi*f_offset*t + phi_data(t) + phi_noise(t)) )
    z(t)^4 = A^4 * exp( j * (8*pi*f_offset*t + 4*phi_data(t) + ...) )

with 4 * phi_data being a multiple of 2*pi and therefore vanishing. The
instantaneous frequency of z^4, divided by four, recovers the carrier
frequency offset -- a direct consequence of transmitter oscillator
inaccuracy, and one of the most established hardware fingerprints in the
RF literature.

CAVEAT TO STATE IN THE WRITE-UP
--------------------------------
Carrier frequency offset conflates two sources: transmitter oscillator
error (hardware, of order 1 kHz) and Doppler shift (geometry, up to
+/- 40 kHz at Iridium's L-band). Doppler is expected to dominate. Removing
it requires computing expected Doppler from satellite position, and 48.6%
of the position fields in this subset failed a physical plausibility check.
Any CFO-driven result must therefore be interpreted with this confound
acknowledged.

USAGE
-----
    python scripts/13_extract_features_v2.py

OUTPUT
------
    outputs/tables/features_v2.csv
        global_index, [26 features], satellite_id
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

TARGET_SATELLITES = [92, 85, 87, 51, 109]
SAMPLE_RATE_HZ    = 25_000_000
N_SEGMENTS        = 4          # sub-windows for the power trajectory

def messages_per_segment() -> int:
    first = sorted(DATA_DIR.glob("ra_sat_*.npy"))[0]
    return int(np.load(first, mmap_mode="r").shape[0])

def extract_v2(i_raw: np.ndarray, q_raw: np.ndarray) -> dict:
    """
    Compute 26 amplitude-invariant, temporally-aware features from one
    message.
    """
    f: dict = {}

    # Every subsequent quantity is then unaffected by how strong the signal
    # arrived, which is what removes the amplitude dominance seen in v1.
    z_raw = i_raw.astype(np.float64) + 1j * q_raw.astype(np.float64)
    rms = np.sqrt(np.mean(np.abs(z_raw) ** 2))
    if rms <= 0 or not np.isfinite(rms):
        rms = 1.0
    z = z_raw / rms
    i = z.real
    q = z.imag
    amp = np.abs(z)
    n = len(z)

    z4 = z ** 4
    # Guard against zeros, which would make the angle undefined
    z4 = np.where(np.abs(z4) > 1e-30, z4, 1e-30)
    phase4 = np.unwrap(np.angle(z4))

    # Instantaneous frequency in Hz. The /4 undoes the fourth power.
    inst_freq = np.diff(phase4) / (2.0 * np.pi) * SAMPLE_RATE_HZ / 4.0

    # Mean carrier frequency offset: oscillator error plus Doppler.
    f["cfo_hz"] = float(np.mean(inst_freq))

    # Short-term spread of the frequency estimate: oscillator instability
    # and phase noise.
    f["cfo_std_hz"] = float(np.std(inst_freq))

    # Robust spread, less sensitive to residual symbol-transition spikes.
    f["cfo_iqr_hz"] = float(np.percentile(inst_freq, 75)
                            - np.percentile(inst_freq, 25))

    # Linear drift of frequency across the burst, in Hz per millisecond.
    # A warming oscillator drifts monotonically; this captures that.
    t_ms = np.arange(len(inst_freq)) / SAMPLE_RATE_HZ * 1e3
    if len(t_ms) > 2 and np.std(t_ms) > 0:
        slope = np.polyfit(t_ms, inst_freq, 1)[0]
    else:
        slope = 0.0
    f["cfo_drift_hz_per_ms"] = float(slope)

    # Shape of the frequency distribution. Asymmetry and heavy tails
    # distinguish smooth drift from impulsive phase disturbance.
    f["cfo_skew"] = float(scipy_stats.skew(inst_freq))
    f["cfo_kurt"] = float(scipy_stats.kurtosis(inst_freq))

    # B. I/Q IMBALANCE  (5 features)
    # An ideal quadrature modulator produces I and Q with equal gain and
    # exactly 90 degrees of separation. Real analogue hardware does not,
    # and the residual imbalance is transmitter-specific.

    std_i = float(np.std(i))
    std_q = float(np.std(q))

    # Gain imbalance in dB. Zero for a perfectly balanced modulator.
    f["iq_gain_imbalance_db"] = float(
        20.0 * np.log10((std_i + 1e-12) / (std_q + 1e-12)))

    # Quadrature phase error. For ideal QPSK, I and Q are uncorrelated;
    # a phase error phi induces correlation of approximately sin(phi), so
    # the error is recovered as arcsin of the sample correlation.
    if std_i > 0 and std_q > 0:
        rho = float(np.corrcoef(i, q)[0, 1])
        rho = float(np.clip(rho, -0.999999, 0.999999))
        f["iq_phase_error_deg"] = float(np.degrees(np.arcsin(rho)))
    else:
        f["iq_phase_error_deg"] = 0.0

    # DC offset: a constant leakage added by the modulator, expressed as a
    # fraction of signal amplitude so it stays amplitude-invariant.
    f["dc_offset_i"]   = float(np.mean(i))
    f["dc_offset_q"]   = float(np.mean(q))
    f["dc_offset_mag"] = float(np.hypot(np.mean(i), np.mean(q)))

    mean_amp = float(np.mean(amp)) + 1e-12

    # Coefficient of variation: envelope ripple relative to its own level,
    # so it does not change with received power.
    f["envelope_cv"] = float(np.std(amp) / mean_amp)

    # Linear trend in the envelope across the burst, normalised. Captures
    # amplifier droop or power-control action during transmission.
    t_norm = np.linspace(0.0, 1.0, n)
    if n > 2:
        f["envelope_slope"] = float(np.polyfit(t_norm, amp, 1)[0] / mean_amp)
    else:
        f["envelope_slope"] = 0.0

    # Peak-to-average power ratio in dB. Scale-invariant by construction.
    inst_power = amp ** 2
    f["papr_db"] = float(10.0 * np.log10(
        (np.max(inst_power) + 1e-30) / (np.mean(inst_power) + 1e-30)))

    # Tailedness of the envelope: sensitive to clipping and compression.
    f["envelope_kurt"] = float(scipy_stats.kurtosis(amp))

    # spread that remains measures how tightly the transmitter holds its
    # constellation -- independent of any fixed rotation between receiver
    # and transmitter.
    angle4 = np.angle(z4)
    # Circular standard deviation: sqrt(-2 ln R), with R the resultant
    # length of the unit phasors. Appropriate for angular data, where the
    # ordinary standard deviation is distorted by wrap-around.
    R = float(np.abs(np.mean(np.exp(1j * angle4))))
    R = min(max(R, 1e-12), 1.0)
    f["constellation_spread_deg"] = float(np.degrees(np.sqrt(-2.0 * np.log(R))))

    # Magnitude dispersion about the mean radius: a proxy for error vector
    # magnitude that requires no symbol decisions.
    f["evm_proxy"] = float(np.std(amp) / mean_amp)

    # Short-term phase jitter: standard deviation of the second difference
    # of unwrapped phase. Differencing twice removes both constant offset
    # and linear drift, isolating high-frequency instability.
    if len(phase4) > 2:
        f["phase_jitter"] = float(np.std(np.diff(phase4, n=2)))
    else:
        f["phase_jitter"] = 0.0

    # structure that any whole-message aggregate discards. Each segment's
    # power is expressed as a ratio to the message mean, so the trajectory
    # is amplitude-invariant while its shape is preserved.
    seg_len = n // N_SEGMENTS
    total_power = float(np.mean(inst_power)) + 1e-30
    for s in range(N_SEGMENTS):
        lo = s * seg_len
        hi = (s + 1) * seg_len if s < N_SEGMENTS - 1 else n
        f[f"seg{s+1}_power_ratio"] = float(np.mean(inst_power[lo:hi]) / total_power)

    # computed on unnormalised samples alongside 13 amplitude-scale features
    # that dominated the model. Recomputed here on the normalised signal so
    # that they sit in a feature set where nothing measures raw power.
    f["skew_i_norm"] = float(scipy_stats.skew(i))
    f["skew_q_norm"] = float(scipy_stats.skew(q))
    f["kurt_i_norm"] = float(scipy_stats.kurtosis(i))
    f["kurt_q_norm"] = float(scipy_stats.kurtosis(q))

    return f

def build_index() -> dict:
    """Group target messages by segment file so each is loaded once."""
    sat_files = sorted(DATA_DIR.glob("ra_sat_*.npy"))
    all_sats = np.concatenate([np.load(f) for f in sat_files])
    per_seg = messages_per_segment()

    target = np.where(np.isin(all_sats, TARGET_SATELLITES))[0]
    print(f"  Total messages:  {len(all_sats):,}")
    print(f"  Target messages: {len(target):,}")

    by_segment: dict = {}
    for g in target:
        by_segment.setdefault(int(g) // per_seg, []).append({
            "global_index": int(g),
            "local_index":  int(g) % per_seg,
            "satellite_id": int(all_sats[g]),
        })
    for seg in sorted(by_segment):
        print(f"    segment {seg}: {len(by_segment[seg]):,} messages")
    return by_segment

def main() -> None:
    print("=" * 72)
    print("Feature extraction v2 - amplitude-invariant, temporally aware")
    print("=" * 72)

    print("\nStep 1 - Indexing target messages...")
    by_segment = build_index()

    print("\nStep 2 - Extracting features...")
    rows = []
    for seg in sorted(by_segment):
        path = DATA_DIR / f"samples_{seg:03d}.npy"
        print(f"  {path.name} ... ", end="", flush=True)
        segment = np.load(path, mmap_mode="r")

        for k, rec in enumerate(by_segment[seg], 1):
            msg = np.array(segment[rec["local_index"]])
            feats = extract_v2(msg[:, 0], msg[:, 1])
            feats["global_index"] = rec["global_index"]
            feats["satellite_id"] = rec["satellite_id"]
            rows.append(feats)
            if k % 500 == 0:
                print(f"{k:,} ", end="", flush=True)

        del segment
        print("done")

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns
                    if c not in ("global_index", "satellite_id")]
    df = df[["global_index"] + feature_cols + ["satellite_id"]]

    print("\nStep 3 - Validating...")
    n_nan = int(df.isna().sum().sum())
    n_inf = int(np.isinf(df[feature_cols].to_numpy()).sum())
    print(f"  Rows:     {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  NaN:      {n_nan}")
    print(f"  Inf:      {n_inf}")

    if n_nan or n_inf:
        print("  Replacing non-finite values with column medians...")
        for c in feature_cols:
            v = df[c].to_numpy(dtype=float)
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = np.nanmedian(v[~bad]) if (~bad).any() else 0.0
                df[c] = v

    print("\n  Per-satellite feature means (first 6 features):")
    preview = df.groupby("satellite_id")[feature_cols[:6]].mean()
    print(preview.to_string(float_format=lambda x: f"{x:11.4g}"))

    out = OUT_TABLES / "features_v2.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Shape: {df.shape}")
    print("\nNext: python scripts/16_compare_v1_v2.py")
    print("=" * 72)

if __name__ == "__main__":
    main()
