"""
04_extract_features.py
======================

INPUT
-----
    data/raw/samples_NNN.npy   raw IQ messages, shape (n, 11000, 2) float32
    data/raw/ra_sat_NNN.npy    satellite ID per message (the label)

OUTPUT
------
    outputs/tables/features.csv
        One row per message. Columns:
        - global_index   position of the message in the full dataset
                         (traceability: lets you go back to the raw signal)
        - satellite_id   the classification label
        - 28 feature columns (defined below, one formula comment each)

THE 28 FEATURES
---------------
Time-domain statistics of I and Q (10):
    mean_I, mean_Q, var_I, var_Q, std_I, std_Q, max_I, max_Q, min_I, min_Q
Distribution shape of I and Q (4):
    skew_I, skew_Q, kurt_I, kurt_Q
Robust statistics of I and Q (4):
    median_I, median_Q, iqr_I, iqr_Q
Cross-channel and power (3):
    signal_power, iq_ratio, iq_correlation
Envelope / temporal (2):
    papr, zero_crossing_rate
Frequency domain, all from ONE shared FFT per message (5):
    fft_mean_magnitude, peak_frequency, spectral_centroid,
    bandwidth, occupied_bandwidth

MEMORY STRATEGY
---------------
Segment files are memory-mapped (mmap_mode="r"), so only the rows
belonging to the five target satellites are ever read from disk.
Peak RAM is one message batch at a time — never a full 880 MB segment.

Run-time: roughly 1-3 minutes for ~6,160 messages on a MacBook Air
(the FFT of 11,000 samples dominates; there is exactly one per message).

USAGE
-----
From the project root:
    python scripts/04_extract_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iq_utils import (
    SAMPLE_RATE_HZ,
    amplitude,
    discover_segments,
    fft_spectrum,
    segment_offsets,
    to_complex,
)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

TARGET_SATELLITES = [92, 85, 87, 51, 109]
OBW_POWER_FRACTION = 0.99

def extract_features(message: np.ndarray) -> dict:
    """
    Compute all 28 features for ONE message of shape (n_samples, 2).

    Every feature has its formula in a comment next to it. If you cannot
    explain a line here to your supervisor, stop and ask before running.
    """
    i = message[:, 0].astype(np.float64)
    q = message[:, 1].astype(np.float64)
    z = to_complex(message)
    amp = amplitude(z)

    f = {}

    f["mean_I"] = np.mean(i)
    f["mean_Q"] = np.mean(q)
    f["var_I"]  = np.var(i)
    f["var_Q"]  = np.var(q)
    f["std_I"]  = np.std(i)
    f["std_Q"]  = np.std(q)
    f["max_I"]  = np.max(i)
    f["max_Q"]  = np.max(q)
    f["min_I"]  = np.min(i)
    f["min_Q"]  = np.min(q)

    # Skewness: third standardised moment, mean((x-mu)^3) / sigma^3.
    #   Asymmetry of the sample distribution — nonzero under asymmetric
    #   amplifier clipping / DC offset.
    # Kurtosis (Fisher): fourth standardised moment minus 3.
    #   0 for Gaussian; clipping lowers it, impulsive noise raises it.
    f["skew_I"] = stats.skew(i)
    f["skew_Q"] = stats.skew(q)
    f["kurt_I"] = stats.kurtosis(i)
    f["kurt_Q"] = stats.kurtosis(q)

    # Median and IQR are outlier-resistant counterparts of mean and std:
    # IQR = 75th percentile - 25th percentile.
    p25_i, p75_i = np.percentile(i, [25, 75])
    p25_q, p75_q = np.percentile(q, [25, 75])
    f["median_I"] = np.median(i)
    f["median_Q"] = np.median(q)
    f["iqr_I"]    = p75_i - p25_i
    f["iqr_Q"]    = p75_q - p25_q

    # Signal power: mean instantaneous power, mean(I^2 + Q^2) = mean(|z|^2).
    f["signal_power"] = np.mean(amp ** 2)
    # I/Q ratio: std_I / std_Q. Exactly 1 for a perfectly balanced
    # transmitter; deviation measures IQ gain imbalance (a classic
    # hardware fingerprint). Guarded against divide-by-zero.
    f["iq_ratio"] = f["std_I"] / f["std_Q"] if f["std_Q"] > 0 else np.nan
    # I/Q correlation: Pearson correlation between I and Q.
    #   corr = cov(I,Q) / (std_I * std_Q).
    # Ideally 0 (I and Q orthogonal); nonzero indicates quadrature
    # skew — another hardware impairment.
    if f["std_I"] > 0 and f["std_Q"] > 0:
        f["iq_correlation"] = np.corrcoef(i, q)[0, 1]
    else:
        f["iq_correlation"] = np.nan

    # PAPR: peak-to-average power ratio, max(|z|^2) / mean(|z|^2).
    #   Sensitive to amplifier saturation (compressed peaks -> lower PAPR).
    f["papr"] = np.max(amp ** 2) / f["signal_power"] if f["signal_power"] > 0 else np.nan
    # Zero-crossing rate of the I channel: fraction of adjacent sample
    # pairs where the sign flips. Related to the dominant frequency
    # content of the in-phase waveform.
    f["zero_crossing_rate"] = np.mean(np.diff(np.signbit(i)) != 0)

    # All five spectral features are computed from a single FFT call.
    # This matters for speed (the FFT is the most expensive operation
    # per message) and for consistency (they all describe one spectrum).
    freqs, mag = fft_spectrum(z, SAMPLE_RATE_HZ)
    power = mag ** 2
    total_power = np.sum(power)

    # Mean FFT magnitude: mean(|X(f)|) over all bins.
    f["fft_mean_magnitude"] = np.mean(mag)
    # Peak frequency: the frequency bin with the largest magnitude —
    # in baseband terms, the residual carrier offset (Doppler +
    # oscillator error), measured in Hz.
    f["peak_frequency"] = freqs[np.argmax(mag)]
    # Spectral centroid: power-weighted mean frequency,
    #   sum(f * P(f)) / sum(P(f)).
    # The "centre of mass" of the spectrum.
    centroid = np.sum(freqs * power) / total_power if total_power > 0 else np.nan
    f["spectral_centroid"] = centroid
    # Bandwidth (RMS): power-weighted standard deviation of frequency
    # around the centroid, sqrt(sum((f - centroid)^2 * P) / sum(P)).
    # How spread out the spectrum is.
    if total_power > 0:
        f["bandwidth"] = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * power) / total_power
        )
    else:
        f["bandwidth"] = np.nan
    # Occupied bandwidth: the frequency span containing the central 99%
    # of total power. Walk the cumulative power across frequency and
    # find where it passes 0.5% and 99.5%.
    if total_power > 0:
        cum = np.cumsum(power) / total_power
        lo_edge = (1.0 - OBW_POWER_FRACTION) / 2.0        # 0.005
        hi_edge = 1.0 - lo_edge                            # 0.995
        f_lo = freqs[np.searchsorted(cum, lo_edge)]
        f_hi = freqs[np.searchsorted(cum, hi_edge)]
        f["occupied_bandwidth"] = f_hi - f_lo
    else:
        f["occupied_bandwidth"] = np.nan

    return f

def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("Feature extraction — Steps 6 + 7")
    print("=" * 70)

    ra_sat_files = discover_segments(DATA_DIR, "ra_sat")
    all_ids = np.concatenate([np.load(f) for f in ra_sat_files])
    target_mask = np.isin(all_ids, TARGET_SATELLITES)
    target_global_indices = np.where(target_mask)[0]
    print(f"Total messages in dataset:        {len(all_ids):,}")
    print(f"Messages from target satellites:  {len(target_global_indices):,}")
    for sat in TARGET_SATELLITES:
        print(f"  Sat {sat:>3d}: {int(np.sum(all_ids == sat)):,} messages")

    samples_files = discover_segments(DATA_DIR, "samples")
    offsets = segment_offsets(samples_files)
    if offsets[-1] != len(all_ids):
        raise ValueError(
            f"Label/sample count mismatch: {len(all_ids):,} labels vs "
            f"{offsets[-1]:,} sample rows. The ra_sat and samples files "
            f"do not describe the same dataset."
        )

    print()
    rows = []
    for seg_idx, seg_path in enumerate(samples_files):
        seg_lo, seg_hi = offsets[seg_idx], offsets[seg_idx + 1]
        in_seg = target_global_indices[
            (target_global_indices >= seg_lo) & (target_global_indices < seg_hi)
        ]
        if len(in_seg) == 0:
            continue

        # Memory-map: only the rows we index are read from disk.
        segment = np.load(seg_path, mmap_mode="r")
        print(f"{seg_path.name}: extracting {len(in_seg):,} messages...",
              end=" ", flush=True)
        t_seg = time.time()

        for g in in_seg:
            local = int(g - seg_lo)
            message = np.asarray(segment[local])
            feats = extract_features(message)
            rows.append({
                "global_index": int(g),
                "satellite_id": int(all_ids[g]),
                **feats,
            })

        print(f"done ({time.time() - t_seg:.1f} s)")

    df = pd.DataFrame(rows)
    n_features = df.shape[1] - 2
    print()
    print(f"Feature matrix: {df.shape[0]:,} rows x {n_features} features")

    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    if nan_count or inf_count:
        print(f"WARNING: {nan_count} NaN and {inf_count} inf values found.")
        print("Affected columns:")
        bad = df.columns[df.isna().any() | np.isinf(
            df.select_dtypes(include=[np.number])).any()]
        for col in bad:
            print(f"  - {col}")
        print("Inspect these before training — do not silently drop them.")
    else:
        print("Validity check: no NaN, no inf. Clean.")

    # Class balance sanity check
    print("\nRows per satellite in features.csv:")
    for sat, count in df["satellite_id"].value_counts().sort_index().items():
        print(f"  Sat {sat:>3d}: {count:,}")

    out_path = OUTPUT_TABLES / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Total time: {time.time() - t_start:.1f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()
