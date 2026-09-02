"""
16_position_sanity.py
=====================

PURPOSE
-------
Rule out the one remaining assumption that could invalidate the whole
project: that `ra_sat` identifies the transmitting satellite and
`ra_lat`/`ra_lon`/`ra_alt` describe that same satellite's position.

WHY THIS MATTERS
----------------
48.6% of decoded positions place the satellite below the receiver's
horizon, which is impossible for a message that was actually received.
Decode confidence does not explain it (Mann-Whitney p = 0.17), and the
values are structured rather than random (`ra_lon` alone classifies at
32.5%), so simple corruption is ruled out.

The remaining possibility is that the position fields do not describe the
transmitting satellite -- for instance if `ra_sat` names a satellite the
message refers to rather than the one broadcasting it. If so, the class
labels would not identify the transmitter and the classification task
would be mislabelled throughout.

THE TEST
--------
Orbital mechanics provide a decisive check. A satellite moves at
approximately 7.5 km/s in low Earth orbit, so between two messages
separated by dt seconds its sub-point can move at most about 7.5 * dt km.

If consecutive messages from one `ra_sat` show ground-track motion
consistent with that limit, the position field describes that satellite's
own trajectory and the labelling is sound. If positions jump erratically,
it does not.

A second check compares implied speed against the orbital velocity: the
distribution should concentrate near 7.5 km/s, not scatter arbitrarily.

USAGE
-----
    python scripts/16_position_sanity.py

OUTPUTS
-------
    outputs/figures/position_sanity.png
    outputs/reports/position_sanity.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET_SATELLITES = [92, 85, 87, 51, 109]
ORBITAL_SPEED_KMS = 7.5          # Iridium, low Earth orbit
EARTH_RADIUS_KM   = 6371.0
MAX_GAP_S         = 600          # messages further apart are different passes

def load_col(name: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{name}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None

def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance along the Earth's surface between two sub-points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def main() -> None:
    print("=" * 72)
    print("Position field sanity check")
    print("=" * 72)

    sat = load_col("ra_sat")
    lat = load_col("ra_lat")
    lon = load_col("ra_lon")
    ts  = load_col("timestamp_global")
    if any(a is None for a in (sat, lat, lon, ts)):
        print("Required columns unavailable.")
        return

    # Timestamps arrive in nanoseconds; convert to seconds.
    ts = ts.astype(float)
    if np.nanmedian(np.abs(ts)) > 1e17:
        ts = ts / 1e9
    if np.nanmax(np.abs(lat)) <= 2.0:
        lat, lon = np.degrees(lat), np.degrees(lon)

    print(f"\n  {len(sat):,} messages, {len(np.unique(sat))} satellites")
    print(f"  Capture spans {(ts.max() - ts.min()) / 3600:.1f} hours")

    all_speeds = []
    per_sat = []

    print(f"\n  {'sat':>5}{'msgs':>8}{'pairs':>8}{'median km/s':>14}"
          f"{'% plausible':>13}")
    print("  " + "-" * 48)

    for s in TARGET_SATELLITES:
        m = (sat == s)
        if m.sum() < 20:
            continue
        order = np.argsort(ts[m])
        t = ts[m][order]
        la = lat[m][order]
        lo = lon[m][order]

        dt = np.diff(t)
        # Consider only consecutive messages within the same pass
        keep = (dt > 0) & (dt < MAX_GAP_S)
        if keep.sum() < 5:
            continue

        dist = great_circle_km(la[:-1][keep], lo[:-1][keep],
                               la[1:][keep],  lo[1:][keep])
        speed = dist / dt[keep]

        # A sub-point cannot move faster than orbital velocity. Allow 50%
        # headroom for timing jitter and quantisation of the position field.
        plausible = float((speed <= ORBITAL_SPEED_KMS * 1.5).mean())

        all_speeds.append(speed)
        per_sat.append({"satellite": int(s), "n_messages": int(m.sum()),
                        "n_pairs": int(keep.sum()),
                        "median_speed": float(np.median(speed)),
                        "frac_plausible": plausible})
        print(f"  {int(s):>5}{int(m.sum()):>8,}{int(keep.sum()):>8,}"
              f"{np.median(speed):>14.2f}{plausible:>12.1%}")

    if not all_speeds:
        print("\n  Not enough within-pass pairs to test.")
        return

    speeds = np.concatenate(all_speeds)
    overall = float((speeds <= ORBITAL_SPEED_KMS * 1.5).mean())
    median_speed = float(np.median(speeds))

    print("\n" + "-" * 72)
    print(f"  Overall median implied speed: {median_speed:.2f} km/s")
    print(f"  Iridium orbital velocity:     {ORBITAL_SPEED_KMS:.2f} km/s")
    print(f"  Pairs within 1.5x orbital velocity: {overall:.1%}")

    if overall > 0.90:
        verdict = ("Positions move at orbital velocity between consecutive "
                   "messages from the same ra_sat. The field describes that "
                   "satellite's own trajectory, and the labelling is sound.")
        print("\n  => LABELLING CONFIRMED.")
    elif overall > 0.60:
        verdict = ("Most transitions are consistent with orbital motion, but "
                   "a substantial minority are not. The field is broadly "
                   "trustworthy with some corrupted entries.")
        print("\n  => MOSTLY CONSISTENT, some bad entries.")
    else:
        verdict = ("Implied speeds are largely inconsistent with orbital "
                   "motion. The position field does not describe the "
                   "trajectory of the satellite named by ra_sat, and the "
                   "meaning of both fields requires re-examination.")
        print("\n  => INCONSISTENT. Investigate before relying on ra_sat.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(np.clip(speeds, 0, 30), bins=80, color="steelblue")
    ax1.axvline(ORBITAL_SPEED_KMS, color="crimson", ls="--", lw=1.5,
                label=f"orbital velocity {ORBITAL_SPEED_KMS} km/s")
    ax1.set_xlabel("Implied sub-point speed (km/s)")
    ax1.set_ylabel("Consecutive message pairs")
    ax1.set_title(f"Implied speed between consecutive messages\n"
                  f"{overall:.1%} within 1.5x orbital velocity",
                  fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Ground track for the satellite with most messages
    best = max(per_sat, key=lambda r: r["n_messages"])
    s = best["satellite"]
    m = (sat == s)
    order = np.argsort(ts[m])
    ax2.scatter(lon[m][order], lat[m][order], s=4, alpha=0.4,
                c=np.arange(m.sum()), cmap="viridis")
    ax2.scatter([-1.2544], [51.7548], marker="*", s=220, color="crimson",
                zorder=5, label="Oxford receiver")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_title(f"Reported ground track, satellite {s}\n"
                  "(colour indicates time order)", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "position_sanity.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    md = f"""# Position field sanity check

## Why this check was performed

48.6% of decoded positions place the satellite below the receiver's
horizon, which cannot occur for a message that was received. Decode
confidence does not explain the discrepancy (Mann-Whitney p = 0.17) and the
values are structured rather than random, so corruption is not a sufficient
explanation.

The remaining possibility is that `ra_lat`/`ra_lon`/`ra_alt` do not
describe the satellite named by `ra_sat`. If that were so, the class labels
would not identify the transmitter and the classification task would be
mislabelled.

## Method

A satellite in low Earth orbit moves at approximately
{ORBITAL_SPEED_KMS} km/s, so between consecutive messages separated by dt
seconds its sub-point can travel at most about {ORBITAL_SPEED_KMS} * dt km.
Implied speeds were computed for consecutive messages from the same
satellite within a single pass (gaps under {MAX_GAP_S} s).

## Results

| Satellite | Messages | Pairs | Median implied speed (km/s) | Within 1.5x orbital velocity |
|----------:|---------:|------:|----------------------------:|-----------------------------:|
"""
    for r in per_sat:
        md += (f"| {r['satellite']} | {r['n_messages']:,} | {r['n_pairs']:,} "
               f"| {r['median_speed']:.2f} | {r['frac_plausible']:.1%} |\n")

    md += f"""
Overall median implied speed: **{median_speed:.2f} km/s** against an orbital
velocity of {ORBITAL_SPEED_KMS} km/s. {overall:.1%} of consecutive pairs
fall within 1.5 times orbital velocity.

## Conclusion

{verdict}
"""
    (OUT_REPORTS / "position_sanity.md").write_text(md)

    print("\n  outputs/figures/position_sanity.png")
    print("  outputs/reports/position_sanity.md")
    print("=" * 72)

if __name__ == "__main__":
    main()
