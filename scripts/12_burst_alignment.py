"""
08_burst_alignment.py
=====================

PURPOSE
-------
Verify that the 11,000-sample capture window is actually filled by the
Iridium burst, and quantify how much of it is noise-only.

WHY THIS MATTERS MORE THAN ANYTHING ELSE IN THE PIPELINE
--------------------------------------------------------
Every one of the 28 features is a statistic computed over the whole
11,000-sample window. If the transmitted burst occupies only part of that
window and the remainder is receiver noise, then every feature is a
weighted blend of signal and noise -- and the blend ratio depends on where
the demodulator happened to trigger, not on the transmitter.

Concretely: `std_I` over a window that is 60% burst and 40% noise is a
different quantity from `std_I` over a window that is 90% burst. Two
messages from the SAME satellite would then differ in every amplitude
feature purely because of trigger timing. That alone is sufficient to
destroy class separability, independently of whether hardware fingerprints
exist in the signal.

SatIQ operates on the message header specifically, not on an arbitrary
window, which is a strong hint that alignment is not automatic.

WHAT THIS SCRIPT DOES
---------------------
  1. Loads a sample of messages and computes each one's amplitude envelope.
  2. Estimates the noise floor from the quietest region of each window.
  3. Detects the burst as the contiguous region above a threshold.
  4. Reports burst start, end, duration, and the fraction of the window
     that is burst vs noise -- and how much these vary between messages.
  5. Recomputes a small set of features on the burst-only region and
     compares their class-discriminative power (ANOVA F) against the same
     features computed on the full window.

Step 5 is the decisive test. If burst-only features have materially higher
F-statistics, then window misalignment was masking real signal and the
feature extraction must be redone on aligned bursts.

USAGE
-----
    python scripts/08_burst_alignment.py

OUTPUTS
-------
    outputs/figures/burst_alignment.png
    outputs/tables/burst_alignment.csv
    outputs/reports/burst_alignment.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET_SATELLITES    = [92, 85, 87, 51, 109]
SAMPLE_RATE_HZ       = 25_000_000
# Derived from the data rather than hardcoded, so the script does not
# silently mis-index if the segment size ever differs.
def _messages_per_segment() -> int:
    first = sorted(DATA_RAW.glob("ra_sat_*.npy"))[0]
    return int(np.load(first, mmap_mode="r").shape[0])

# How many messages per satellite to analyse. 150 x 5 = 750 messages is
# plenty to characterise the burst structure and is fast to process.
MESSAGES_PER_SAT = 150

# Envelope smoothing window, in samples. At 25 MS/s, 50 samples = 2 us,
# short enough to preserve burst edges but long enough to suppress the
# sample-to-sample fluctuation of QPSK symbol transitions.
SMOOTH_SAMPLES = 50

# Burst threshold as a fraction of the way from noise floor to peak.
# 0.25 is deliberately permissive: we want to find the burst EDGES, and
# a high threshold would clip the rise and fall.
BURST_THRESHOLD_FRAC = 0.25

# How many example envelopes to keep per satellite for its detail figure.
EXAMPLES_PER_SATELLITE = 6

def select_messages() -> dict[int, list[dict]]:
    """
    Choose MESSAGES_PER_SAT messages for each target satellite, grouped by
    which segment file they live in so each 880 MB file is loaded once.
    """
    sat_files = sorted(DATA_RAW.glob("ra_sat_*.npy"))
    all_sats = np.concatenate([np.load(f) for f in sat_files])
    per_seg = _messages_per_segment()

    rng = np.random.default_rng(42)
    by_segment: dict[int, list[dict]] = {}

    for sat in TARGET_SATELLITES:
        idx = np.where(all_sats == sat)[0]
        if len(idx) > MESSAGES_PER_SAT:
            idx = rng.choice(idx, MESSAGES_PER_SAT, replace=False)
        for g in idx:
            seg = int(g) // per_seg
            by_segment.setdefault(seg, []).append({
                "global_index": int(g),
                "local_index":  int(g) % per_seg,
                "satellite_id": int(sat),
            })

    return by_segment

def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average via convolution, same length as input."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

def detect_burst(i: np.ndarray, q: np.ndarray) -> dict:
    """
    Locate the transmitted burst within one capture window.

    Method:
      1. amplitude envelope = sqrt(I^2 + Q^2), smoothed
      2. noise floor       = 10th percentile of the envelope
                             (assumes at least some of the window is quiet;
                              if the burst fills the window this will simply
                              sit just below the burst level, and the
                              detected burst will span nearly everything --
                              which is the answer we want to know)
      3. peak              = 90th percentile (robust to spikes)
      4. threshold         = noise + 0.25 * (peak - noise)
      5. burst             = first to last sample above threshold

    Returns start/end indices, duration, and the burst fraction of window.
    """
    envelope = smooth(np.sqrt(i**2 + q**2), SMOOTH_SAMPLES)
    n = len(envelope)

    noise_floor = float(np.percentile(envelope, 10))
    peak        = float(np.percentile(envelope, 90))
    dynamic_range = peak - noise_floor

    # If peak and floor are nearly equal there is no detectable burst
    # structure -- either the window is all burst or all noise.
    if dynamic_range <= 1e-12 or peak <= 0:
        return {"start": 0, "end": n - 1, "duration": n,
                "burst_fraction": 1.0, "snr_db": np.nan,
                "detected": False}

    threshold = noise_floor + BURST_THRESHOLD_FRAC * dynamic_range
    above = np.where(envelope > threshold)[0]

    if len(above) == 0:
        return {"start": 0, "end": n - 1, "duration": n,
                "burst_fraction": 1.0, "snr_db": np.nan,
                "detected": False}

    start, end = int(above[0]), int(above[-1])
    duration = end - start + 1

    # Crude SNR: burst power over noise-floor power, in dB
    snr_db = float(10 * np.log10((peak**2) / (noise_floor**2 + 1e-20)))

    return {"start": start, "end": end, "duration": duration,
            "burst_fraction": duration / n, "snr_db": snr_db,
            "detected": True}

def quick_features(i: np.ndarray, q: np.ndarray) -> dict:
    """
    A small, representative subset of the 28 features -- enough to test
    whether restricting to the burst changes discriminative power, without
    recomputing everything.
    """
    amp = np.sqrt(i**2 + q**2)
    return {
        "std_I":       float(np.std(i)),
        "std_Q":       float(np.std(q)),
        "signal_power": float(np.mean(i**2 + q**2)),
        "papr":        float(10 * np.log10(
                           (np.max(amp**2) + 1e-20) /
                           (np.mean(amp**2) + 1e-20))),
        "kurt_I":      float(scipy_stats.kurtosis(i)),
        "iq_corr":     float(np.corrcoef(i, q)[0, 1])
                       if np.std(i) > 0 and np.std(q) > 0 else 0.0,
    }

def anova_f_per_feature(df: pd.DataFrame,
                        feature_cols: list[str],
                        label_col: str) -> dict[str, float]:
    """
    One-way ANOVA F-statistic per feature, across satellite classes.

    F measures between-class variance over within-class variance. A larger
    F means the feature separates the classes better. F near 1 means the
    feature varies as much within a satellite as between satellites -- i.e.
    it carries no class information.
    """
    out = {}
    for col in feature_cols:
        groups = [g[col].to_numpy() for _, g in df.groupby(label_col)]
        groups = [g[np.isfinite(g)] for g in groups]
        if any(len(g) < 2 for g in groups):
            out[col] = np.nan
            continue
        try:
            out[col] = float(scipy_stats.f_oneway(*groups).statistic)
        except Exception:
            out[col] = np.nan
    return out

def plot_alignment(records: pd.DataFrame,
                   examples: list[tuple[int, np.ndarray, dict]],
                   path: Path) -> None:
    examples = sorted(examples, key=lambda e: e[0])
    fig = plt.figure(figsize=(19, 10))
    n_ex = min(len(examples), 5)
    gs = fig.add_gridspec(3, 2 * max(n_ex, 1))

    # Row 1: three example envelopes with detected burst marked
    for k, (sat, envelope, burst) in enumerate(examples[:n_ex]):
        ax = fig.add_subplot(gs[0, 2*k:2*k+2])
        t = np.arange(len(envelope)) / SAMPLE_RATE_HZ * 1e6
        ax.plot(t, envelope, lw=0.6, color="steelblue")
        if burst["detected"]:
            ax.axvspan(t[burst["start"]], t[burst["end"]],
                       color="orange", alpha=0.25, label="detected burst")
        ax.set_title(f"Sat {sat} - burst fills "
                     f"{burst['burst_fraction']:.0%} of window", fontsize=10)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)

    # Row 2 left: distribution of burst fraction
    ax = fig.add_subplot(gs[1, 0:n_ex])
    ax.hist(records["burst_fraction"], bins=40, color="seagreen")
    ax.set_xlabel("Fraction of window occupied by burst")
    ax.set_ylabel("Messages")
    ax.set_title("How much of the window is signal?", fontweight="bold")
    ax.grid(alpha=0.3)

    # Row 2 right: distribution of burst start position
    ax = fig.add_subplot(gs[1, n_ex:2*n_ex])
    ax.hist(records["start"], bins=40, color="darkorange")
    ax.set_xlabel("Burst start (sample index)")
    ax.set_ylabel("Messages")
    ax.set_title("Is the burst consistently positioned?", fontweight="bold")
    ax.grid(alpha=0.3)

    # Row 3 left: burst fraction per satellite
    ax = fig.add_subplot(gs[2, 0:n_ex])
    sats = sorted(records["satellite_id"].unique())
    ax.boxplot([records.loc[records["satellite_id"] == s, "burst_fraction"]
                for s in sats], tick_labels=[str(s) for s in sats])
    ax.set_xlabel("Satellite")
    ax.set_ylabel("Burst fraction")
    ax.set_title("Burst fraction per satellite", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Row 3 right: burst fraction vs a scale feature
    ax = fig.add_subplot(gs[2, n_ex:2*n_ex])
    ax.scatter(records["burst_fraction"], records["full_std_I"],
               s=8, alpha=0.4, color="crimson")
    ok = records["burst_fraction"].notna() & records["full_std_I"].notna()
    if ok.sum() > 10:
        r = np.corrcoef(records.loc[ok, "burst_fraction"],
                        records.loc[ok, "full_std_I"])[0, 1]
        ax.set_title(f"std_I vs burst fraction (r = {r:+.3f})\n"
                     "strong correlation => feature tracks trigger timing",
                     fontweight="bold", fontsize=10)
    ax.set_xlabel("Burst fraction")
    ax.set_ylabel("std_I (full window)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

def plot_single_satellite(sat: int,
                          sat_records: pd.DataFrame,
                          sat_examples: list[tuple[np.ndarray, dict]],
                          path: Path) -> None:
    """
    Detail figure for one satellite.

    The question this answers: is the burst consistently positioned within
    the capture window for THIS transmitter? If burst start varies widely
    between messages from the same satellite, then every amplitude feature
    for that satellite is partly a measure of trigger timing, and the
    within-class scatter that swamps class separation has a known cause.
    """
    n_ex = len(sat_examples)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # Rows 1-2: up to six example envelopes
    for k, (envelope, burst) in enumerate(sat_examples[:6]):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        t = np.arange(len(envelope)) / SAMPLE_RATE_HZ * 1e6
        ax.plot(t, envelope, lw=0.6, color="steelblue")
        if burst["detected"]:
            ax.axvspan(t[burst["start"]], t[burst["end"]],
                       color="orange", alpha=0.25)
        ax.set_title(f"message {k+1}: burst = {burst['burst_fraction']:.0%} "
                     f"of window, starts at sample {burst['start']:,}",
                     fontsize=9)
        ax.set_xlabel("Time (us)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    bf = sat_records["burst_fraction"]
    st = sat_records["start"]

    # Row 3 left: burst fraction for this satellite
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(bf, bins=30, color="seagreen")
    ax.axvline(bf.median(), color="black", ls="--", lw=1)
    ax.set_xlabel("Burst fraction of window", fontsize=9)
    ax.set_ylabel("Messages", fontsize=9)
    ax.set_title(f"median {bf.median():.3f}, std {bf.std():.3f}",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)

    # Row 3 middle: burst start position
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(st, bins=30, color="darkorange")
    ax.axvline(st.median(), color="black", ls="--", lw=1)
    ax.set_xlabel("Burst start (sample index)", fontsize=9)
    ax.set_ylabel("Messages", fontsize=9)
    ax.set_title(f"median {st.median():.0f}, std {st.std():.0f} samples",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)

    # Row 3 right: does a scale feature track burst fraction?
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(bf, sat_records["full_std_I"], s=10, alpha=0.5, color="crimson")
    ok = bf.notna() & sat_records["full_std_I"].notna()
    title = "std_I vs burst fraction"
    if ok.sum() > 10 and bf[ok].std() > 0:
        r = np.corrcoef(bf[ok], sat_records.loc[ok, "full_std_I"])[0, 1]
        title += f"  (r = {r:+.3f})"
    ax.set_xlabel("Burst fraction", fontsize=9)
    ax.set_ylabel("std_I (full window)", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Satellite {sat} - burst alignment "
                 f"({len(sat_records):,} messages analysed)",
                 fontsize=14, fontweight="bold")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    print("=" * 70)
    print("Burst alignment diagnostic")
    print("=" * 70)

    print("\nStep 1 - Selecting messages...")
    by_segment = select_messages()
    total = sum(len(v) for v in by_segment.values())
    print(f"  {total:,} messages across {len(by_segment)} segment files")

    print("\nStep 2 - Loading segments and detecting bursts...")
    records = []
    examples = []
    seen_examples: set[int] = set()
    # Envelopes kept per satellite for the individual detail figures.
    # 6 x 5 x 11,000 floats is about 2.6 MB, so this is cheap.
    per_sat_examples: dict[int, list[tuple[np.ndarray, dict]]] = {}

    for seg in sorted(by_segment.keys()):
        path = DATA_RAW / f"samples_{seg:03d}.npy"
        print(f"  {path.name}... ", end="", flush=True)
        segment = np.load(path, mmap_mode="r")   # memory-map: only read rows we need

        for rec in by_segment[seg]:
            msg = np.array(segment[rec["local_index"]])
            i, q = msg[:, 0], msg[:, 1]

            burst = detect_burst(i, q)
            full  = quick_features(i, q)

            # Features on the burst region only
            if burst["detected"] and burst["duration"] > 100:
                bi = i[burst["start"]:burst["end"] + 1]
                bq = q[burst["start"]:burst["end"] + 1]
                bfeat = quick_features(bi, bq)
            else:
                bfeat = full

            row = {"satellite_id": rec["satellite_id"],
                   "window_length": len(i), **burst}
            row.update({f"full_{k}": v for k, v in full.items()})
            row.update({f"burst_{k}": v for k, v in bfeat.items()})
            records.append(row)

            # One example per satellite, so the panels are comparable.
            # Taking the first three messages processed would show three
            # examples of the same satellite, since messages are queued
            # satellite by satellite.
            sat = rec["satellite_id"]
            store = per_sat_examples.setdefault(sat, [])
            if len(store) < EXAMPLES_PER_SATELLITE:
                env = smooth(np.sqrt(i**2 + q**2), SMOOTH_SAMPLES)
                store.append((env, burst))
                if sat not in seen_examples:
                    examples.append((sat, env, burst))
                    seen_examples.add(sat)

        del segment
        print("done")

    df = pd.DataFrame(records)

    print("\nStep 3 - Burst geometry:")
    n_win = int(df["window_length"].iloc[0])
    bf = df["burst_fraction"]
    print(f"  Window length:         {n_win:,} samples "
          f"({n_win / SAMPLE_RATE_HZ * 1e6:.0f} us)")
    print(f"  Burst fraction: median {bf.median():.3f}   "
          f"mean {bf.mean():.3f}   std {bf.std():.3f}")
    print(f"                  min    {bf.min():.3f}   max  {bf.max():.3f}")
    print(f"  Burst start:    median {df['start'].median():.0f}   "
          f"std {df['start'].std():.0f} samples")
    print(f"  Detected in {int(df['detected'].sum()):,} / {len(df):,} messages")
    print("\n  Per satellite:")
    print(f"    {'sat':>5}{'n':>7}{'burst frac':>13}{'std':>8}"
          f"{'start median':>15}{'start std':>11}")
    for s in sorted(df["satellite_id"].unique()):
        d = df[df["satellite_id"] == s]
        print(f"    {int(s):>5}{len(d):>7,}{d['burst_fraction'].median():>13.3f}"
              f"{d['burst_fraction'].std():>8.3f}"
              f"{d['start'].median():>15,.0f}{d['start'].std():>11,.0f}")

    if bf.median() > 0.95:
        verdict = ("Burst fills essentially the whole window. Window "
                   "misalignment is NOT a confound.")
    elif bf.std() < 0.05:
        verdict = (f"Burst occupies {bf.median():.0%} of the window but very "
                   "consistently. A fixed offset affects all messages "
                   "equally, so it is unlikely to destroy separability.")
    else:
        verdict = (f"Burst occupies {bf.median():.0%} of the window and "
                   f"VARIES between messages (std {bf.std():.3f}). Every "
                   "amplitude feature is partly a measure of trigger "
                   "timing. This is a genuine confound.")
    print(f"\n  => {verdict}")

    print("\nStep 4 - Class-discriminative power, full window vs burst only:")
    names = ["std_I", "std_Q", "signal_power", "papr", "kurt_I", "iq_corr"]
    f_full  = anova_f_per_feature(df, [f"full_{n}"  for n in names], "satellite_id")
    f_burst = anova_f_per_feature(df, [f"burst_{n}" for n in names], "satellite_id")

    print(f"  {'feature':<16}{'F (full)':>10}{'F (burst)':>12}{'change':>10}")
    print("  " + "-" * 48)
    comparison = []
    for n in names:
        a = f_full.get(f"full_{n}", np.nan)
        b = f_burst.get(f"burst_{n}", np.nan)
        change = (b - a) / a * 100 if (np.isfinite(a) and a > 0) else np.nan
        comparison.append({"feature": n, "F_full_window": a,
                           "F_burst_only": b, "pct_change": change})
        print(f"  {n:<16}{a:>10.2f}{b:>12.2f}{change:>9.0f}%")

    mean_full  = np.nanmean([c["F_full_window"] for c in comparison])
    mean_burst = np.nanmean([c["F_burst_only"]  for c in comparison])
    print("  " + "-" * 48)
    print(f"  {'mean':<16}{mean_full:>10.2f}{mean_burst:>12.2f}"
          f"{(mean_burst - mean_full) / mean_full * 100:>9.0f}%")

    if mean_burst > mean_full * 1.5:
        conclusion = ("Trimming to the burst SUBSTANTIALLY increases "
                      "discriminative power. Feature extraction should be "
                      "redone on aligned bursts.")
    elif mean_burst > mean_full * 1.1:
        conclusion = ("Trimming gives a modest improvement. Worth redoing "
                      "extraction, but it will not change the headline result.")
    else:
        conclusion = ("Trimming does NOT improve discriminative power. "
                      "Window alignment is not what is limiting the "
                      "classifier; the features themselves carry no "
                      "transmitter information.")
    print(f"\n  => {conclusion}")

    # For reference: F ~ 1 means no separation at all
    print(f"\n  (F = 1.0 would mean no class separation whatsoever.)")

    print("\nStep 5 - Writing outputs...")
    df.to_csv(OUT_TABLES / "burst_alignment.csv", index=False)
    plot_alignment(df, examples, OUT_FIGURES / "burst_alignment.png")

    # One detail figure per satellite
    per_sat_dir = OUT_FIGURES / "burst_per_satellite"
    per_sat_dir.mkdir(parents=True, exist_ok=True)
    for sat in sorted(df["satellite_id"].unique()):
        sat_df = df[df["satellite_id"] == sat]
        plot_single_satellite(int(sat), sat_df,
                              per_sat_examples.get(int(sat), []),
                              per_sat_dir / f"burst_sat_{int(sat)}.png")
    print(f"  outputs/figures/burst_per_satellite/ "
          f"({df['satellite_id'].nunique()} figures)")

    md = f"""# Burst alignment diagnostic

## Why this check was performed

All 28 features are statistics computed over an 11,000-sample
({n_win / SAMPLE_RATE_HZ * 1e6:.0f} us) capture window. If the transmitted
burst occupies only part of that window, every feature is a blend of signal
and receiver noise, with the blend ratio set by demodulator trigger timing
rather than by the transmitter. Two messages from the same satellite would
then differ in every amplitude feature for reasons unrelated to hardware.

## Burst geometry

Measured on {len(df):,} messages across {df['satellite_id'].nunique()} satellites.

| Quantity | Value |
|----------|------:|
| Window length | {n_win:,} samples ({n_win / SAMPLE_RATE_HZ * 1e6:.0f} us) |
| Burst fraction (median) | {bf.median():.3f} |
| Burst fraction (std across messages) | {bf.std():.3f} |
| Burst fraction (min / max) | {bf.min():.3f} / {bf.max():.3f} |
| Burst start (median sample) | {df['start'].median():.0f} |
| Burst start (std) | {df['start'].std():.0f} samples |

**{verdict}**

## Does trimming to the burst improve class separation?

One-way ANOVA F-statistic per feature, computed on the full window and on
the burst region only. F is the ratio of between-class to within-class
variance; F = 1 indicates no class separation.

| Feature | F (full window) | F (burst only) | Change |
|---------|----------------:|---------------:|-------:|
"""
    for c in comparison:
        md += (f"| `{c['feature']}` | {c['F_full_window']:.2f} "
               f"| {c['F_burst_only']:.2f} | {c['pct_change']:+.0f}% |\n")
    md += (f"| **mean** | **{mean_full:.2f}** | **{mean_burst:.2f}** "
           f"| **{(mean_burst - mean_full) / mean_full * 100:+.0f}%** |\n")

    md += f"""
**{conclusion}**

## Interpretation

This diagnostic distinguishes two competing explanations for the
chance-level baseline:

1. *The features are fine, but the extraction window is misaligned*, so
   signal is diluted by noise in a message-dependent way.
2. *The features themselves carry no transmitter-specific information.*

The F-statistic comparison above discriminates between them directly: if
explanation 1 held, restricting the computation to the burst would raise
the F-statistics materially. The measured result is reported above.
"""
    (OUT_REPORTS / "burst_alignment.md").write_text(md)

    print("  outputs/tables/burst_alignment.csv")
    print("  outputs/figures/burst_alignment.png")
    print("  outputs/reports/burst_alignment.md")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
