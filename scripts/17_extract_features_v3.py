"""
16_extract_features_v3.py
=========================

PURPOSE
-------
Extract power-amplifier nonlinearity features, which neither v1 nor v2
contains.

MOTIVATION
----------
A March 2026 theoretical study of satellite RF fingerprint limits reports
that IQ imbalance under certain modulation schemes may carry insufficient
identifying information, while power-amplifier nonlinearities are more
reliable, and obtains AUC 0.934 on Iridium using a feature-weighted method.

That finding maps directly onto results already obtained here. The v2
feature set contained `iq_gain_imbalance_db` and `iq_phase_error_deg` and
did not improve on v1 (McNemar p = 0.71). The study offers a theoretical
reason why, and names the alternative that has not been extracted at any
point in this project: amplifier nonlinearity.

WHAT AMPLIFIER NONLINEARITY LOOKS LIKE IN A RECEIVED SIGNAL
------------------------------------------------------------
A power amplifier driven near saturation distorts in two coupled ways:

  AM/AM  the output amplitude ceases to be proportional to the input,
         compressing the envelope at high drive levels

  AM/PM  the output PHASE becomes dependent on the input AMPLITUDE, so
         high-amplitude samples are rotated differently from low-amplitude
         ones

AM/PM is the more useful of the two for fingerprinting. In an ideal
transmitter, amplitude and phase are independent; any systematic coupling
between them is introduced by the amplifier, and its precise form depends
on that individual device's bias point and semiconductor characteristics.

Nonlinearity also produces intermodulation products that fall outside the
occupied bandwidth. The resulting spectral regrowth, measured as adjacent
channel power ratio, is a further amplifier signature.

None of these quantities requires knowledge of the transmitted data, and
all are computed here as ratios, angles or normalised quantities so that
they are unaffected by received signal strength -- avoiding the amplitude
dominance that limited the v1 set.

USAGE
-----
    python scripts/16_extract_features_v3.py

OUTPUT
------
    outputs/tables/features_v3.csv
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
N_AMP_BINS        = 4          # amplitude quartiles for AM/PM measurement


def messages_per_segment() -> int:
    first = sorted(DATA_DIR.glob("ra_sat_*.npy"))[0]
    return int(np.load(first, mmap_mode="r").shape[0])


# --- FEATURE EXTRACTION ---------------------------------------------------
def extract_v3(i_raw: np.ndarray, q_raw: np.ndarray) -> dict:
    """Compute 20 amplifier-nonlinearity features from one message."""
    f: dict = {}

    # Normalise to unit average power. Every feature below is a ratio, an
    # angle or a normalised quantity, so none depends on received strength.
    z = i_raw.astype(np.float64) + 1j * q_raw.astype(np.float64)
    rms = np.sqrt(np.mean(np.abs(z) ** 2))
    if rms <= 0 or not np.isfinite(rms):
        rms = 1.0
    z = z / rms
    amp = np.abs(z)
    n = len(z)

    # Remove QPSK data modulation by the fourth-power method, leaving a
    # residual phase that reflects the transmitter rather than the data.
    kernel = np.ones(16) / 16.0
    z_s = (np.convolve(z.real, kernel, mode="same")
           + 1j * np.convolve(z.imag, kernel, mode="same"))
    z4 = z_s ** 4
    z4 = np.where(np.abs(z4) > 1e-30, z4, 1e-30)
    residual_phase = np.angle(z4) / 4.0          # radians
    amp_s = np.abs(z_s)

    # =====================================================================
    # A. AM/PM CONVERSION  (6 features)
    # =====================================================================
    # In a linear transmitter, phase does not depend on amplitude. Any
    # systematic dependence is introduced by the amplifier, and its shape
    # is characteristic of the individual device.

    ok = np.isfinite(residual_phase) & np.isfinite(amp_s) & (amp_s > 0)
    a_ok = amp_s[ok]
    p_ok = residual_phase[ok]

    if len(a_ok) > 100 and np.std(a_ok) > 0:
        # Correlation between amplitude and residual phase. Zero for an
        # ideal amplifier.
        f["ampm_correlation"] = float(np.corrcoef(a_ok, p_ok)[0, 1])

        # Slope of the phase-versus-amplitude relationship, in degrees per
        # unit of normalised amplitude. This is the conventional AM/PM
        # conversion coefficient.
        slope = np.polyfit(a_ok, p_ok, 1)[0]
        f["ampm_slope_deg"] = float(np.degrees(slope))

        # Mean residual phase within each amplitude quartile. A linear
        # amplifier gives the same value in every quartile; a compressing
        # one does not. Reporting the quartiles separately captures
        # curvature that a single slope would miss.
        edges = np.percentile(a_ok, np.linspace(0, 100, N_AMP_BINS + 1))
        bin_means = []
        for b in range(N_AMP_BINS):
            m = (a_ok >= edges[b]) & (a_ok <= edges[b + 1])
            bin_means.append(float(np.mean(p_ok[m])) if m.sum() > 10 else np.nan)
        bin_means = np.array(bin_means)

        f["ampm_q1_deg"] = float(np.degrees(bin_means[0]))
        f["ampm_q4_deg"] = float(np.degrees(bin_means[-1]))
        # Total phase excursion across the amplitude range
        f["ampm_range_deg"] = float(np.degrees(
            np.nanmax(bin_means) - np.nanmin(bin_means)))
        # Curvature: second-order term in the phase-amplitude relationship
        f["ampm_curvature"] = float(np.polyfit(a_ok, p_ok, 2)[0]) \
            if len(a_ok) > 200 else 0.0
    else:
        for k in ("ampm_correlation", "ampm_slope_deg", "ampm_q1_deg",
                  "ampm_q4_deg", "ampm_range_deg", "ampm_curvature"):
            f[k] = 0.0

    # =====================================================================
    # B. AM/AM COMPRESSION  (5 features)
    # =====================================================================
    # An amplifier driven towards saturation clips the envelope peaks. The
    # complementary cumulative distribution of instantaneous power shows
    # this as a shortened tail relative to an undistorted signal.

    inst_power = amp ** 2
    mean_power = float(np.mean(inst_power)) + 1e-30
    power_ratio = inst_power / mean_power

    # Probability that instantaneous power exceeds the mean by 1, 3, 6, 9 dB
    for db in (1, 3, 6, 9):
        threshold = 10 ** (db / 10.0)
        f[f"ccdf_{db}db"] = float(np.mean(power_ratio > threshold))

    # Measured PAPR relative to the value expected for an undistorted
    # pulse-shaped QPSK signal (approximately 5 dB). A markedly lower value
    # indicates the amplifier is compressing the peaks.
    papr_db = 10.0 * np.log10(np.max(power_ratio) + 1e-30)
    f["papr_deficit_db"] = float(papr_db - 5.0)

    # =====================================================================
    # C. SPECTRAL REGROWTH  (5 features)
    # =====================================================================
    # Intermodulation products created by nonlinearity fall outside the
    # occupied bandwidth. The power that appears in adjacent bands is a
    # direct measure of amplifier linearity.

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(z))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / SAMPLE_RATE_HZ))
    total = float(np.sum(spectrum)) + 1e-30

    # Locate the occupied band adaptively as the narrowest contiguous span
    # around the peak containing 90% of the total power.
    peak_idx = int(np.argmax(spectrum))
    cumulative = 0.0
    half_width = 0
    while cumulative < 0.90 * total and half_width < n // 4:
        half_width += 1
        lo = max(peak_idx - half_width, 0)
        hi = min(peak_idx + half_width + 1, n)
        cumulative = float(np.sum(spectrum[lo:hi]))

    in_lo, in_hi = max(peak_idx - half_width, 0), min(peak_idx + half_width + 1, n)
    in_band = float(np.sum(spectrum[in_lo:in_hi])) + 1e-30

    # Adjacent channels: bands of equal width immediately either side
    w = in_hi - in_lo
    lower = float(np.sum(spectrum[max(in_lo - w, 0):in_lo]))
    upper = float(np.sum(spectrum[in_hi:min(in_hi + w, n)]))

    f["acpr_lower_db"] = float(10.0 * np.log10((lower + 1e-30) / in_band))
    f["acpr_upper_db"] = float(10.0 * np.log10((upper + 1e-30) / in_band))
    f["acpr_mean_db"]  = float((f["acpr_lower_db"] + f["acpr_upper_db"]) / 2)

    # Asymmetry between the two adjacent bands. Symmetric regrowth arises
    # from amplitude compression alone; asymmetry indicates that amplitude
    # and phase distortion are interacting.
    f["acpr_asymmetry_db"] = float(f["acpr_upper_db"] - f["acpr_lower_db"])

    # Occupied bandwidth as a fraction of the capture bandwidth. A wider
    # occupied band for the same protocol implies greater regrowth.
    f["occupied_fraction"] = float(w / n)

    # =====================================================================
    # D. CONSTELLATION QUALITY BY AMPLITUDE  (4 features)
    # =====================================================================
    # Stratifying constellation tightness by amplitude separates distortion
    # that grows with drive level from distortion that does not. A linear
    # transmitter shows similar spread in every stratum.

    if len(a_ok) > 200:
        edges = np.percentile(a_ok, np.linspace(0, 100, N_AMP_BINS + 1))
        angle4_ok = np.angle(z4)[ok]
        for b in range(N_AMP_BINS):
            m = (a_ok >= edges[b]) & (a_ok <= edges[b + 1])
            if m.sum() > 20:
                # Circular standard deviation, appropriate for angles
                R = float(np.abs(np.mean(np.exp(1j * angle4_ok[m]))))
                R = min(max(R, 1e-12), 1.0)
                f[f"const_spread_q{b+1}_deg"] = float(
                    np.degrees(np.sqrt(-2.0 * np.log(R))))
            else:
                f[f"const_spread_q{b+1}_deg"] = 0.0
    else:
        for b in range(N_AMP_BINS):
            f[f"const_spread_q{b+1}_deg"] = 0.0

    return f


# --- INDEX ---------------------------------------------------------------
def build_index() -> dict:
    sat_files = sorted(DATA_DIR.glob("ra_sat_*.npy"))
    all_sats = np.concatenate([np.load(f) for f in sat_files])
    per_seg = messages_per_segment()
    target = np.where(np.isin(all_sats, TARGET_SATELLITES))[0]
    print(f"  Target messages: {len(target):,}")

    by_segment: dict = {}
    for g in target:
        by_segment.setdefault(int(g) // per_seg, []).append({
            "global_index": int(g),
            "local_index":  int(g) % per_seg,
            "satellite_id": int(all_sats[g]),
        })
    return by_segment


def main() -> None:
    print("=" * 72)
    print("Feature extraction v3 - power-amplifier nonlinearity")
    print("=" * 72)

    print("\nStep 1 - Indexing...")
    by_segment = build_index()

    print("\nStep 2 - Extracting...")
    rows = []
    for seg in sorted(by_segment):
        path = DATA_DIR / f"samples_{seg:03d}.npy"
        print(f"  {path.name} ... ", end="", flush=True)
        segment = np.load(path, mmap_mode="r")
        for k, rec in enumerate(by_segment[seg], 1):
            msg = np.array(segment[rec["local_index"]])
            feats = extract_v3(msg[:, 0], msg[:, 1])
            feats["global_index"] = rec["global_index"]
            feats["satellite_id"] = rec["satellite_id"]
            rows.append(feats)
            if k % 500 == 0:
                print(f"{k:,} ", end="", flush=True)
        del segment
        print("done")

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("global_index", "satellite_id")]
    df = df[["global_index"] + cols + ["satellite_id"]]

    print("\nStep 3 - Validating...")
    n_bad = int((~np.isfinite(df[cols].to_numpy())).sum())
    print(f"  Rows: {len(df):,}   Features: {len(cols)}   Non-finite: {n_bad}")
    if n_bad:
        for c in cols:
            v = df[c].to_numpy(dtype=float)
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = np.nanmedian(v[~bad]) if (~bad).any() else 0.0
                df[c] = v

    # One-way ANOVA per feature: a quick indication of which, if any,
    # separate the classes before any model is fitted.
    print("\n  Class separation per feature (ANOVA F, 1.0 = none):")
    fstats = []
    for c in cols:
        groups = [g[c].to_numpy() for _, g in df.groupby("satellite_id")]
        try:
            fstats.append((c, float(scipy_stats.f_oneway(*groups).statistic)))
        except Exception:
            fstats.append((c, np.nan))
    for c, s in sorted(fstats, key=lambda t: -(t[1] if np.isfinite(t[1]) else 0))[:8]:
        print(f"    {c:<26}{s:>8.2f}")

    out = OUT_TABLES / "features_v3.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Shape: {df.shape}")
    print("\nNext: python scripts/21_openset_and_domain.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
