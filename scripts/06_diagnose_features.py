"""
06_diagnose_features.py
=======================

WHY THIS SCRIPT EXISTS
----------------------
05_train_classifiers.py produced chance-level accuracy (~21-24%) for
every model. There are two possible explanations, and they lead to
completely different next steps:

  A) A BUG - most likely label misalignment: samples_NNN.npy row i does
     not actually correspond to ra_sat_NNN.npy entry i, so every feature
     vector is paired with an unrelated label. Chance accuracy is the
     signature of this bug.

  B) A FINDING - the 28 global-statistics features genuinely carry no
     per-satellite information, because whole-burst averages wash out
     the subtle structure (phase noise, transients, constellation
     distortion) where hardware fingerprints live.

This script runs four independent checks to decide between A and B.

CHECK 1 - ALIGNMENT (the decisive one).
    The dataset ships a 'level' metadata column: received signal
    strength per message, recorded at capture time. Our extracted
    signal_power measures the same physical quantity from the raw IQ.
    If row ordering is consistent across columns, the two MUST be
    strongly correlated. Strong correlation -> alignment confirmed,
    bug ruled out. Near-zero correlation -> misalignment bug found.

CHECK 2 - FEATURE DEGENERACY.
    Are the features near-constant across messages? Prints the spread
    (coefficient of variation) of every feature. If features barely
    vary at all, no classifier could use them.

CHECK 3 - DISCRIMINATIVE POWER (ANOVA F-test).
    For each feature: does its mean differ between satellites more
    than within satellites? (sklearn f_classif). High F -> the feature
    separates classes. All-low F with confirmed alignment -> finding B.

CHECK 4 - VISUAL: PCA of the feature space, coloured by satellite.
    If clusters exist, you'll see them. If it's one undifferentiated
    blob, that is what chance-level accuracy looks like.

USAGE
-----
From the project root:
    python scripts/06_diagnose_features.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV   = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("Feature diagnostics — bug vs finding")
    print("=" * 70)

    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns
                    if c not in ("global_index", "satellite_id")]
    X = df[feature_cols].to_numpy()
    y = df["satellite_id"].to_numpy()
    g = df["global_index"].to_numpy()

    # ─── CHECK 1: ALIGNMENT via the 'level' metadata column ─────────────
    print("\nCHECK 1 — Alignment: dataset 'level' vs extracted signal_power")
    level_files = sorted(DATA_DIR.glob("level_*.npy"))
    if not level_files:
        print("  No level_*.npy files found — skipping (cannot run this check).")
    else:
        all_levels = np.concatenate([np.load(f) for f in level_files])
        msg_levels = all_levels[g].astype(np.float64)
        log_power = np.log10(df["signal_power"].to_numpy())  # level is dB-like
        rho, pval = spearmanr(msg_levels, log_power)
        print(f"  Spearman correlation (level vs log10 signal_power): "
              f"rho = {rho:.3f}  (p = {pval:.2e}, n = {len(g):,})")
        if abs(rho) > 0.5:
            print("  -> STRONG correlation: row alignment CONFIRMED.")
            print("     The metadata and the IQ rows describe the same messages.")
        elif abs(rho) > 0.2:
            print("  -> Moderate correlation: alignment probably OK, but look")
            print("     at the scatter figure before concluding.")
        else:
            print("  -> NEAR-ZERO correlation: LIKELY MISALIGNMENT BUG.")
            print("     Do not interpret classifier results until resolved.")
        # Scatter for the eyeball test
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(msg_levels, log_power, s=3, alpha=0.3, color="steelblue")
        ax.set_xlabel("Dataset 'level' (metadata, recorded at capture)")
        ax.set_ylabel("log10(signal_power) (extracted from raw IQ)")
        ax.set_title(f"Alignment check — Spearman rho = {rho:.3f}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURES / "alignment_check.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ─── CHECK 2: FEATURE DEGENERACY ─────────────────────────────────────
    # Coefficient of variation = std / |mean| (guarded near zero-mean
    # features by falling back to plain std). Near-zero spread means the
    # feature is effectively a constant.
    print("\nCHECK 2 — Feature spread across all messages")
    stds = X.std(axis=0)
    means = np.abs(X.mean(axis=0))
    n_dead = 0
    for name, s, m in zip(feature_cols, stds, means):
        rel = s / m if m > 1e-12 else np.inf
        if s == 0 or (np.isfinite(rel) and rel < 1e-3):
            n_dead += 1
            print(f"  NEAR-CONSTANT: {name:<22s} std={s:.3e} mean={m:.3e}")
    if n_dead == 0:
        print("  No near-constant features — the features do vary "
              "across messages.")
    else:
        print(f"  {n_dead} near-constant feature(s) found.")

    # ─── CHECK 3: ANOVA F-test per feature ───────────────────────────────
    # F = between-class variance / within-class variance. Under the null
    # (feature tells you nothing about the satellite), F ~= 1.
    print("\nCHECK 3 — Per-feature class separation (ANOVA F, sorted)")
    F, p = f_classif(X, y)
    anova = (pd.DataFrame({"feature": feature_cols, "F": F, "p": p})
             .sort_values("F", ascending=False).reset_index(drop=True))
    anova.to_csv(OUTPUT_REPORTS / "feature_anova.csv", index=False)
    for _, row in anova.head(10).iterrows():
        print(f"  {row.feature:<22s} F = {row.F:8.2f}   p = {row.p:.2e}")
    print(f"  ... median F across all 28 features: {np.median(F):.2f}")
    print("  (F near 1 = no class information in that feature;")
    print("   with ~6,000 messages, genuinely useful features show F in")
    print("   the tens to hundreds.)")

    # ─── CHECK 4: PCA scatter ────────────────────────────────────────────
    print("\nCHECK 4 — PCA of the standardised 28-D feature space")
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    print(f"  Variance explained by 2 components: "
          f"{ev[0]:.1%} + {ev[1]:.1%} = {ev.sum():.1%}")
    fig, ax = plt.subplots(figsize=(8, 7))
    for sat in np.unique(y):
        mask = y == sat
        ax.scatter(X2[mask, 0], X2[mask, 1], s=4, alpha=0.4,
                   label=f"Sat {sat}")
    ax.legend(markerscale=3)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} of variance)")
    ax.set_title("Feature space (PCA), coloured by satellite")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "feature_space_pca.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nOutputs written:")
    print(f"  outputs/figures/alignment_check.png")
    print(f"  outputs/figures/feature_space_pca.png")
    print(f"  outputs/reports/feature_anova.csv")
    print("=" * 70)
    print("HOW TO READ THE RESULT")
    print("  Check 1 strong + Check 3 all-low  -> finding: these features")
    print("      genuinely don't separate satellites. Next step is better")
    print("      features, and the chapter narrative is 'why global")
    print("      statistics fail'.")
    print("  Check 1 near-zero                 -> bug: fix alignment first,")
    print("      then re-run 04 and 05. Do not interpret anything yet.")
    print("=" * 70)


if __name__ == "__main__":
    main()
