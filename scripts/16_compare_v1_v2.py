"""
14_compare_v1_v2.py
===================

PURPOSE
-------
Test whether the v2 feature set recovers transmitter information that v1
could not, using the same evaluation protocol throughout so the comparison
is fair.

FOUR MODELS ARE COMPARED
------------------------
    chance          majority-class baseline
    v1 (28)         original hand-crafted features
    v2 (26)         amplitude-invariant, temporally aware features
    v1 + v2 (54)    both sets combined

WHAT WOULD CONSTITUTE SUCCESS
-----------------------------
v1 reached 24.8% accuracy against 20.8% chance, McNemar p = 0.12 -- not
distinguishable from guessing. For v2 to represent progress it must beat
chance by a paired test, not merely post a higher point estimate. Every
comparison below therefore uses McNemar's test, which considers only the
cases where two models disagree and is the correct test when both are
evaluated on the same test set.

INTERPRETING A CFO-DRIVEN RESULT
--------------------------------
If v2 succeeds and `cfo_hz` is the dominant feature, the result requires
care. Carrier frequency offset combines transmitter oscillator error
(hardware, of order 1 kHz) with Doppler shift (geometry, up to +/- 40 kHz
at L-band), and Doppler is expected to dominate. The script therefore also
regresses `cfo_hz` against elevation angle: a strong association would
indicate the model is keying on orbital geometry rather than on oscillator
characteristics, which has the same authentication weakness already
identified for the receiver-side metadata model.

USAGE
-----
    python scripts/14_compare_v1_v2.py

Requires scripts 03 and 13 to have been run.

OUTPUTS
-------
    outputs/tables/v1_v2_comparison.csv
    outputs/tables/v2_feature_importance.csv
    outputs/figures/v1_v2_comparison.png
    outputs/reports/v1_v2_comparison.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
V1_CSV       = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
V2_CSV       = PROJECT_ROOT / "outputs" / "tables" / "features_v2.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
N_BOOTSTRAP   = 2000
NON_FEATURE   = {"sample_id", "global_index", "index",
                 "Unnamed: 0", "satellite_id"}

RECEIVER_LAT, RECEIVER_LON, RECEIVER_ALT = 51.7548, -1.2544, 60.0

def load_and_merge() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge v1 and v2 feature tables on global_index."""
    if not V2_CSV.exists():
        raise FileNotFoundError(f"Run script 13 first: {V2_CSV}")

    v1 = pd.read_csv(V1_CSV)
    v2 = pd.read_csv(V2_CSV)

    v1_features = [c for c in v1.columns if c not in NON_FEATURE]
    v2_features = [c for c in v2.columns if c not in NON_FEATURE]

    merged = v1.merge(v2.drop(columns=["satellite_id"]),
                      on="global_index", how="inner",
                      suffixes=("", "_v2dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_v2dup")]]

    print(f"  v1: {len(v1):,} rows, {len(v1_features)} features")
    print(f"  v2: {len(v2):,} rows, {len(v2_features)} features")
    print(f"  merged on global_index: {len(merged):,} rows")
    return merged, v1_features, v2_features

def clean(X: np.ndarray) -> np.ndarray:
    X = X.astype(float).copy()
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0
    return X

def add_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """Elevation angle, for the Doppler-confound check (same method as 06/09)."""
    def col(name):
        files = sorted(DATA_RAW.glob(f"{name}_*.npy"))
        return np.concatenate([np.load(f) for f in files]) if files else None

    rows = df["global_index"].to_numpy(dtype=int)
    lat_a, lon_a, alt_a = col("ra_lat"), col("ra_lon"), col("ra_alt")
    if any(a is None for a in (lat_a, lon_a, alt_a)):
        return df
    if rows.max() >= len(lat_a):
        return df

    lat, lon, radial = lat_a[rows], lon_a[rows], alt_a[rows]
    if np.nanmax(np.abs(lat)) <= 2.0:
        lat, lon = np.degrees(lat), np.degrees(lon)

    med = float(np.nanmedian(radial))
    if 6_000 < med < 10_000:
        radius_m = radial * 1000.0
    elif 6e6 < med < 1e7:
        radius_m = radial
    elif 100 < med < 2_000:
        radius_m = radial * 1000.0 + 6_371_000.0
    else:
        radius_m = radial + 6_371_000.0

    lat0, lon0 = np.radians(RECEIVER_LAT), np.radians(RECEIVER_LON)
    A, F = 6378137.0, 1/298.257223563
    E2 = F * (2 - F)
    N = A / np.sqrt(1 - E2 * np.sin(lat0) ** 2)
    rx = ((N + RECEIVER_ALT) * np.cos(lat0) * np.cos(lon0),
          (N + RECEIVER_ALT) * np.cos(lat0) * np.sin(lon0),
          (N * (1 - E2) + RECEIVER_ALT) * np.sin(lat0))

    la, lo = np.radians(lat), np.radians(lon)
    dx = radius_m * np.cos(la) * np.cos(lo) - rx[0]
    dy = radius_m * np.cos(la) * np.sin(lo) - rx[1]
    dz = radius_m * np.sin(la)               - rx[2]
    east  = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = (-np.sin(lat0) * np.cos(lon0) * dx
             - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz)
    up    = ( np.cos(lat0) * np.cos(lon0) * dx
             + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz)
    elev = np.degrees(np.arctan2(up, np.sqrt(east ** 2 + north ** 2)))
    elev[elev < 0] = np.nan
    df["elevation_deg"] = elev
    return df

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact McNemar p-value for two classifiers on the same test set."""
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    if n01 + n10 == 0:
        return 1.0
    return float(scipy_stats.binomtest(n10, n01 + n10, 0.5).pvalue)

def bootstrap_ci(correct: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    m = len(correct)
    s = [correct[rng.integers(0, m, m)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def main() -> None:
    print("=" * 72)
    print("v1 vs v2 feature comparison")
    print("=" * 72)

    print("\nStep 1 - Loading...")
    df, v1_feats, v2_feats = load_and_merge()
    df = add_elevation(df)
    y = df["satellite_id"].to_numpy()

    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    def run(feature_names: list[str], label: str) -> dict:
        X = clean(df[feature_names].to_numpy())
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf",   RandomForestClassifier(n_estimators=300,
                                             random_state=RANDOM_SEED,
                                             n_jobs=-1)),
        ]).fit(X[idx_tr], y_tr)
        pred = model.predict(X[idx_te])
        correct = (pred == y_te)
        lo, hi = bootstrap_ci(correct)
        return {"label": label, "n_features": len(feature_names),
                "accuracy": float(correct.mean()),
                "macro_f1": float(f1_score(y_te, pred, average="macro",
                                           zero_division=0)),
                "ci_low": lo, "ci_high": hi,
                "correct": correct, "model": model,
                "features": feature_names}

    print("\nStep 2 - Training...")
    dummy_pred = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(idx_tr), 1)), y_tr).predict(np.zeros((len(idx_te), 1)))
    correct_chance = (dummy_pred == y_te)
    lo, hi = bootstrap_ci(correct_chance)
    chance = {"label": "chance (majority class)", "n_features": 0,
              "accuracy": float(correct_chance.mean()),
              "macro_f1": float(f1_score(y_te, dummy_pred, average="macro",
                                         zero_division=0)),
              "ci_low": lo, "ci_high": hi, "correct": correct_chance}

    results = [chance,
               run(v1_feats, "v1 (hand-crafted)"),
               run(v2_feats, "v2 (amplitude-invariant)"),
               run(v1_feats + v2_feats, "v1 + v2 combined")]

    print(f"\n  {'model':<28}{'feat':>6}{'acc':>9}{'95% CI':>20}{'F1':>9}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['label']:<28}{r['n_features']:>6}{r['accuracy']:>9.4f}"
              f"   [{r['ci_low']:.4f}, {r['ci_high']:.4f}]{r['macro_f1']:>9.4f}")

    print("\nStep 3 - Paired tests (McNemar) against chance:")
    for r in results[1:]:
        p = mcnemar(chance["correct"], r["correct"])
        r["p_vs_chance"] = p
        verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {r['label']:<28} p = {p:.4f}   {verdict}")

    p_v1_v2 = mcnemar(results[1]["correct"], results[2]["correct"])
    print(f"\n  v2 vs v1 directly:           p = {p_v1_v2:.4f}   "
          f"{'SIGNIFICANT' if p_v1_v2 < 0.05 else 'not significant'}")

    print("\nStep 4 - v2 feature importance:")
    v2_res = results[2]
    forest = v2_res["model"].named_steps["clf"]
    imp = (pd.DataFrame({"feature": v2_feats,
                         "importance": forest.feature_importances_})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    for _, r in imp.head(8).iterrows():
        print(f"    {r['feature']:<26}{r['importance']:.4f}")

    print("\nStep 5 - Is cfo_hz measuring hardware or geometry?")
    if "cfo_hz" in df.columns and "elevation_deg" in df.columns:
        ok = df["cfo_hz"].notna() & df["elevation_deg"].notna()
        if ok.sum() > 50:
            r_el = float(np.corrcoef(df.loc[ok, "cfo_hz"],
                                     df.loc[ok, "elevation_deg"])[0, 1])
            print(f"  cfo_hz vs elevation: r = {r_el:+.3f}  "
                  f"(R^2 = {r_el**2:.3f}, n = {int(ok.sum()):,})")
        # How much of cfo_hz is explained by satellite identity alone?
        vals = df["cfo_hz"].to_numpy(float)
        fin = np.isfinite(vals)
        grand = vals[fin].mean()
        ss_tot = float(np.sum((vals[fin] - grand) ** 2))
        ss_bet = sum(len(vals[fin & (y == s)]) *
                     (vals[fin & (y == s)].mean() - grand) ** 2
                     for s in np.unique(y) if (fin & (y == s)).any())
        print(f"  cfo_hz explained by satellite identity: "
              f"R^2 = {ss_bet / ss_tot if ss_tot else np.nan:.3f}")
        print("  Per-satellite mean cfo_hz:")
        for s in np.unique(y):
            m = fin & (y == s)
            print(f"    Sat {int(s):>3d}: {vals[m].mean():>12,.1f} Hz  "
                  f"(std {vals[m].std():>10,.1f})")

    print("\nStep 6 - Writing outputs...")
    table = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("correct", "model", "features")}
                          for r in results])
    table.to_csv(OUT_TABLES / "v1_v2_comparison.csv", index=False)
    imp.to_csv(OUT_TABLES / "v2_feature_importance.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    labels = [r["label"] for r in results]
    accs   = [r["accuracy"] for r in results]
    errs   = [[r["accuracy"] - r["ci_low"] for r in results],
              [r["ci_high"] - r["accuracy"] for r in results]]
    colours = ["grey"] + ["steelblue", "seagreen", "darkorange"]
    ax1.bar(range(len(results)), accs, yerr=errs, capsize=5, color=colours)
    ax1.axhline(chance["accuracy"], color="crimson", ls="--", lw=1.2,
                label="chance")
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Feature set comparison", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    top = imp.head(15).iloc[::-1]
    ax2.barh(top["feature"], top["importance"], color="seagreen")
    ax2.set_xlabel("Random Forest importance")
    ax2.set_title("v2 feature importance (top 15)", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "v1_v2_comparison.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    md = f"""# v1 vs v2 feature comparison

## Design

The v2 feature set was constructed to address two diagnosed weaknesses in
v1: amplitude dominance (13 of 28 features had R² above 0.98 against the
receiver's amplitude estimate) and loss of temporal structure (all v1
features were aggregates, unchanged by reordering the samples).

All models use the same train/test split, the same Random Forest settings,
and the same chance baseline, so differences are attributable to the
features alone.

| Model | Features | Accuracy | 95% CI | Macro F1 | p vs chance |
|-------|---------:|---------:|:------:|---------:|------------:|
"""
    for r in results:
        p = f"{r['p_vs_chance']:.4f}" if "p_vs_chance" in r else "-"
        md += (f"| {r['label']} | {r['n_features']} | {r['accuracy']:.4f} "
               f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
               f"| {r['macro_f1']:.4f} | {p} |\n")

    md += f"""
Direct comparison of v2 against v1 (McNemar): p = {p_v1_v2:.4f}.

## v2 feature importance

| Rank | Feature | Importance |
|-----:|---------|-----------:|
"""
    for i, r in imp.head(10).iterrows():
        md += f"| {i+1} | `{r['feature']}` | {r['importance']:.4f} |\n"

    md += """
## Interpreting a CFO-driven result

Carrier frequency offset combines two sources. Transmitter oscillator error
is a hardware property, typically of order 1 kHz. Doppler shift is a
geometric property, reaching approximately +/- 40 kHz at Iridium's L-band
for a satellite in low Earth orbit. Doppler is therefore expected to
dominate the measured offset.

If `cfo_hz` ranks highly and is strongly associated with elevation angle,
the model is keying on orbital geometry rather than on oscillator
characteristics. That has the same authentication weakness already
identified for the receiver-side metadata model: an adversary transmitting
from a plausible position produces a plausible offset, without imitating
any hardware property.

Separating the two would require subtracting predicted Doppler computed
from satellite ephemeris. In this subset 48.6% of reported positions failed
a physical plausibility check, so that correction cannot be applied
reliably here and is left as future work.
"""
    (OUT_REPORTS / "v1_v2_comparison.md").write_text(md)

    print("  outputs/tables/v1_v2_comparison.csv")
    print("  outputs/tables/v2_feature_importance.csv")
    print("  outputs/figures/v1_v2_comparison.png")
    print("  outputs/reports/v1_v2_comparison.md")
    print("=" * 72)

if __name__ == "__main__":
    main()
