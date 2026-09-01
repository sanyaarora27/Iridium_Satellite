"""
05_feature_importance.py
========================

PURPOSE
-------

Three analyses:

  A. RANDOM FOREST FEATURE IMPORTANCE  (Step 10, required)
     Which of the 28 features does the model lean on most?

  B. CHANNEL-DOMINANCE ANALYSIS  (the diagnostic)
     For each feature, how much of its variance is explained by the
     receiver's estimated signal level alone? A feature with a high R^2
     against `level` is largely measuring "how far away was the satellite",
     not "which transmitter sent this".

  C. THE METADATA-ONLY CONTROL  (the decisive test)
     Train a classifier using ONLY channel metadata (level, noise,
     altitude, centre frequency) and no signal features whatsoever.

     If metadata alone matches the accuracy of the 28-feature model, then
     the hand-crafted features contribute nothing beyond channel geometry.
     This distinguishes "my features failed" from "my features measured
     the channel, not the hardware" -- a much more precise claim.

WHY THIS MATTERS
----------------
Step 10 asks, for each important feature, whether it relates to hardware
imperfections, signal power, channel effects, or Doppler/elevation.
Analysis B answers that question with a measured number instead of an
assertion, and analysis C tests the answer directly.

USAGE
-----
From the project root:
    python scripts/05_feature_importance.py

OUTPUTS
-------
    outputs/figures/feature_importance.png
    outputs/figures/channel_dominance.png
    outputs/tables/feature_importance.csv
    outputs/tables/channel_dominance.csv
    outputs/reports/feature_analysis.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---------------------------------------------------------------
TEST_FRACTION = 0.20
RANDOM_SEED   = 42

# Columns in features.csv that are not physical signal properties.
NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}

# Metadata columns to pull from data/raw/. Each is stored as a series of
# segment files named e.g. level_000.npy ... level_004.npy
METADATA_COLUMNS = ["level", "noise", "ra_alt", "center_frequency"]


# --- STEP 1: LOAD FEATURES -----------------------------------------------
def load_features() -> tuple[pd.DataFrame, list[str]]:
    """Read features.csv and identify the genuine feature columns."""
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_names


# --- STEP 2: ATTACH CHANNEL METADATA -------------------------------------
def load_metadata_column(column: str) -> np.ndarray | None:
    """
    Load one metadata column by concatenating its segment files in order.

    The Zenodo dataset stores each column as `{column}_{segment}.npy`.
    Concatenating them in sorted order reproduces the original message
    ordering, so position i in the result corresponds to global_index i.
    """
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    if not files:
        return None
    return np.concatenate([np.load(f) for f in files])


def attach_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Join channel metadata onto the feature table using global_index.

    global_index is the message's position in the full concatenated
    dataset. It is NOT valid as a model input (it leaks capture order),
    but it is exactly the right key for aligning metadata to feature rows.
    """
    index_col = next((c for c in ("global_index", "sample_id")
                      if c in df.columns), None)
    if index_col is None:
        raise KeyError("features.csv has no global_index / sample_id column, "
                       "so metadata cannot be aligned.")

    row_indices = df[index_col].to_numpy(dtype=int)
    attached = []

    for column in METADATA_COLUMNS:
        full = load_metadata_column(column)
        if full is None:
            print(f"    {column:<18s} not found in data/raw -- skipping")
            continue
        if row_indices.max() >= len(full):
            print(f"    {column:<18s} length mismatch -- skipping")
            continue
        df[f"meta_{column}"] = full[row_indices]
        attached.append(f"meta_{column}")
        print(f"    {column:<18s} attached")

    return df, attached


# --- STEP 3: RANDOM FOREST FEATURE IMPORTANCE ----------------------------
def compute_feature_importance(X: np.ndarray,
                               y: np.ndarray,
                               feature_names: list[str]) -> pd.DataFrame:
    """
    Fit a Random Forest and read off its impurity-based feature importances.

    IMPORTANT CAVEAT for the write-up: these importances describe which
    features the forest USED, not which features are genuinely informative.
    When overall accuracy is at chance, high importance means "the model
    split on this a lot while still failing" -- it ranks the features by
    how much apparent structure they offered, not by real predictive value.
    Impurity importance is also known to favour high-cardinality continuous
    features regardless of usefulness.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    forest = RandomForestClassifier(n_estimators=200,
                                    random_state=RANDOM_SEED,
                                    n_jobs=-1)
    forest.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, forest.predict(X_test))

    # Standard deviation of importance across the individual trees gives
    # an error bar: a large spread means the trees disagree about whether
    # the feature matters at all.
    importances = forest.feature_importances_
    spread = np.std([t.feature_importances_ for t in forest.estimators_],
                    axis=0)

    table = pd.DataFrame({
        "feature":     feature_names,
        "importance":  importances,
        "importance_std": spread,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    table.attrs["accuracy"] = accuracy
    return table


# --- STEP 4: CHANNEL-DOMINANCE ANALYSIS ----------------------------------
def compute_channel_dominance(df: pd.DataFrame,
                              feature_names: list[str],
                              level_column: str) -> pd.DataFrame:
    """
    For every feature, measure how much of its variation is explained by
    the receiver's signal level alone.

    For a univariate linear regression, R^2 equals the squared Pearson
    correlation, so we compute it directly. We also report Spearman rho,
    which captures monotonic relationships that are not straight lines
    (many RF quantities scale with level non-linearly).

    Reading the result:
        R^2 near 1.0  -> the feature is essentially a restatement of how
                         strong the received signal was. It carries channel
                         and geometry information, not transmitter identity.
        R^2 near 0.0  -> the feature varies independently of signal level.
                         It MAY carry hardware information (not guaranteed).
    """
    level = df[level_column].to_numpy(dtype=float)
    rows = []

    for name in feature_names:
        values = df[name].to_numpy(dtype=float)

        # Guard against constant or degenerate columns
        ok = np.isfinite(values) & np.isfinite(level)
        if ok.sum() < 10 or np.std(values[ok]) == 0 or np.std(level[ok]) == 0:
            rows.append({"feature": name, "r2_vs_level": np.nan,
                         "pearson_r": np.nan, "spearman_rho": np.nan})
            continue

        pearson_r  = float(np.corrcoef(values[ok], level[ok])[0, 1])
        spearman   = float(scipy_stats.spearmanr(values[ok], level[ok]).statistic)

        rows.append({
            "feature":      name,
            "r2_vs_level":  pearson_r ** 2,   # R^2 of a linear fit
            "pearson_r":    pearson_r,
            "spearman_rho": spearman,
        })

    return (pd.DataFrame(rows)
            .sort_values("r2_vs_level", ascending=False)
            .reset_index(drop=True))


# --- STEP 5: THE METADATA-ONLY CONTROL -----------------------------------
def metadata_only_control(df: pd.DataFrame,
                          metadata_columns: list[str],
                          y: np.ndarray) -> dict:
    """
    Train a Random Forest on channel metadata ALONE -- no signal features.

    This is the decisive test. If a model that never sees the IQ signal
    performs as well as the 28-feature model, then those 28 features are
    adding nothing beyond what the channel geometry already reveals.
    """
    X = df[metadata_columns].to_numpy(dtype=float)

    # Metadata may contain NaNs; replace with column medians so the
    # comparison is not distorted by dropped rows.
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=200,
                                         random_state=RANDOM_SEED,
                                         n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    return {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "columns":  metadata_columns,
    }


# --- STEP 6: PLOTS -------------------------------------------------------
def plot_feature_importance(table: pd.DataFrame, path: Path) -> None:
    """Horizontal bar chart of all features, most important at the top."""
    fig, ax = plt.subplots(figsize=(9, 9))
    order = table.iloc[::-1]                      # reverse for top-down
    ax.barh(order["feature"], order["importance"],
            xerr=order["importance_std"],
            color="steelblue", error_kw={"alpha": 0.4, "lw": 0.8})
    ax.set_xlabel("Random Forest impurity importance")
    ax.set_title(f"Feature importance (overall accuracy "
                 f"{table.attrs['accuracy']:.1%})", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_channel_dominance(dominance: pd.DataFrame,
                           importance: pd.DataFrame,
                           path: Path) -> None:
    """
    Two panels:
      Left  - R^2 of each feature against signal level, sorted.
      Right - importance plotted against R^2. If the features the model
              relies on are also the ones explained by signal level, the
              points trend upward to the right, and the model is leaning
              on channel information.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    d = dominance.dropna(subset=["r2_vs_level"]).iloc[::-1]
    colours = ["crimson" if v > 0.5 else "steelblue" for v in d["r2_vs_level"]]
    ax1.barh(d["feature"], d["r2_vs_level"], color=colours)
    ax1.axvline(0.5, color="black", ls="--", lw=1, alpha=0.6)
    ax1.set_xlabel(r"$R^2$ of feature explained by signal level")
    ax1.set_title(
        "Feature dependence on receiver-reported signal level",
        fontweight="bold"
            )
    ax1.set_xlim(0, 1)
    ax1.grid(axis="x", alpha=0.3)

    merged = importance.merge(dominance, on="feature").dropna(
        subset=["r2_vs_level"])
    ax2.scatter(merged["r2_vs_level"], merged["importance"],
                s=55, color="darkorange", edgecolor="black", lw=0.5)
    for _, r in merged.iterrows():
        ax2.annotate(r["feature"], (r["r2_vs_level"], r["importance"]),
                     fontsize=6.5, alpha=0.75,
                     xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel(r"$R^2$ explained by signal level")
    ax2.set_ylabel("Random Forest importance")
    ax2.set_title(
        "Feature importance versus received-level dependence",
        fontweight="bold"
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- STEP 7: REPORT ------------------------------------------------------
def write_report(importance: pd.DataFrame,
                 dominance: pd.DataFrame,
                 control: dict | None,
                 full_accuracy: float,
                 chance: float,
                 path: Path) -> None:
    md = f"""# Feature importance and channel-dominance analysis

## A. Random Forest feature importance (Task list Step 10)

Overall Random Forest accuracy on the 28 hand-crafted features:
**{full_accuracy:.1%}** against a chance level of {chance:.1%}.

**Interpretation caveat.** Because overall accuracy is at or near chance,
these importances describe which features the forest *split on*, not which
features are genuinely informative. Impurity-based importance also favours
continuous, high-cardinality features irrespective of predictive value.
The ranking below should therefore be read as "where the model looked",
and interpreted together with Section B.

### Top 10 features

| Rank | Feature | Importance | R² explained by signal level | Likely source |
|-----:|---------|-----------:|-----------------------------:|---------------|
"""
    merged = importance.merge(dominance, on="feature", how="left")
    for i, r in merged.head(10).iterrows():
        r2 = r["r2_vs_level"]
        if pd.isna(r2):
            source, r2s = "unknown", "n/a"
        elif r2 > 0.5:
            source, r2s = "channel / signal power", f"{r2:.3f}"
        elif r2 > 0.2:
            source, r2s = "mixed channel + other", f"{r2:.3f}"
        else:
            source, r2s = "not signal-level driven", f"{r2:.3f}"
        md += (f"| {i+1} | `{r['feature']}` | {r['importance']:.4f} "
               f"| {r2s} | {source} |\n")

    valid = dominance.dropna(subset=["r2_vs_level"])
    n_high = int((valid["r2_vs_level"] > 0.5).sum())
    n_tot  = len(valid)
    top    = valid.iloc[0] if len(valid) else None

    md += f"""

## B. Channel-dominance analysis

Each feature was regressed against `level`, the receiver's estimated
signal strength. `level` depends on satellite range, elevation angle and
atmospheric path -- that is, on **geometry at the moment of capture**, not
on transmitter hardware.

- Features whose variance is **majority-explained** by signal level
  (R² > 0.5): **{n_high} of {n_tot}**
"""
    if top is not None:
        md += (f"- Most channel-dominated feature: `{top['feature']}` "
               f"(R² = {top['r2_vs_level']:.3f}, "
               f"Spearman ρ = {top['spearman_rho']:.3f})\n")

    md += """
A feature with high R² against `level` is largely a restatement of how
strong the received signal was. Two messages from the *same* satellite at
different elevations will differ more in these features than two messages
from *different* satellites at similar elevations -- which is precisely the
condition under which per-message classification fails.
"""

    if control is not None:
        gap = full_accuracy - control["accuracy"]
        md += f"""

## C. Metadata-only control experiment

A Random Forest was trained using **only** channel metadata
({', '.join(c.replace('meta_', '') for c in control['columns'])}) and no
signal-derived features at all.

| Model input | Accuracy |
|-------------|---------:|
| 28 hand-crafted signal features | {full_accuracy:.1%} |
| Channel metadata only (no signal) | {control['accuracy']:.1%} |
| Chance (majority class) | {chance:.1%} |

"""
        if abs(gap) < 0.02:
            md += """**The two are equivalent.** A model that never observes the
IQ waveform performs as well as one built on 28 signal-derived features.
The hand-crafted feature set therefore contributes no transmitter-specific
information beyond what channel geometry already supplies. Whatever small
margin exists above chance is attributable to capture geometry, not to
hardware fingerprints.
"""
        elif gap > 0:
            md += f"""The signal features outperform metadata alone by
{gap:.1%}. This margin is the upper bound on any genuinely signal-derived
contribution, and it should be checked against the confidence intervals in
the model comparison before being treated as real.
"""
        else:
            md += f"""Channel metadata alone **outperforms** the 28 signal
features by {abs(gap):.1%}. This indicates the signal features are not
merely uninformative but noisier proxies for the same underlying geometry.
"""

    md += """

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
"""

    with open(path, "w") as f:
        f.write(md)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Feature importance and channel-dominance analysis")
    print("=" * 70)

    # 1. Features
    print("\nStep 1 - Loading features...")
    df, feature_names = load_features()
    y = df["satellite_id"].to_numpy()
    X = df[feature_names].to_numpy(dtype=float)
    chance = float(np.max(np.bincount(pd.factorize(y)[0])) / len(y))
    print(f"  Messages: {len(df):,}   Features: {len(feature_names)}")
    print(f"  Chance level: {chance:.1%}")

    # 2. Metadata
    print("\nStep 2 - Attaching channel metadata from data/raw...")
    df, metadata_columns = attach_metadata(df)

    # 3. Importance
    print("\nStep 3 - Computing Random Forest feature importance...")
    importance = compute_feature_importance(X, y, feature_names)
    full_accuracy = importance.attrs["accuracy"]
    print(f"  Random Forest accuracy: {full_accuracy:.4f}")
    print("  Top 5 features:")
    for i, r in importance.head(5).iterrows():
        print(f"    {i+1}. {r['feature']:<22s} {r['importance']:.4f}")

    # 4. Channel dominance
    dominance = None
    if "meta_level" in df.columns:
        print("\nStep 4 - Channel-dominance analysis (features vs level)...")
        dominance = compute_channel_dominance(df, feature_names, "meta_level")
        valid = dominance.dropna(subset=["r2_vs_level"])
        print(f"  Features with R^2 > 0.5 against level: "
              f"{int((valid['r2_vs_level'] > 0.5).sum())} / {len(valid)}")
        print("  Most channel-dominated:")
        for _, r in valid.head(5).iterrows():
            print(f"    {r['feature']:<22s} R^2={r['r2_vs_level']:.3f}  "
                  f"rho={r['spearman_rho']:+.3f}")
    else:
        print("\nStep 4 - SKIPPED (level metadata unavailable)")

    # 5. Metadata-only control
    control = None
    if metadata_columns:
        print("\nStep 5 - Metadata-only control experiment...")
        control = metadata_only_control(df, metadata_columns, y)
        print(f"  Signal features ({len(feature_names)}): {full_accuracy:.4f}")
        print(f"  Metadata only  ({len(metadata_columns)}): {control['accuracy']:.4f}")
        print(f"  Chance:                        {chance:.4f}")
        if abs(full_accuracy - control['accuracy']) < 0.02:
            print("  => EQUIVALENT: signal features add nothing beyond channel geometry.")
    else:
        print("\nStep 5 - SKIPPED (no metadata attached)")

    # 6. Outputs
    print("\nStep 6 - Writing outputs...")
    importance.to_csv(OUT_TABLES / "feature_importance.csv", index=False)
    plot_feature_importance(importance, OUT_FIGURES / "feature_importance.png")
    print("  outputs/tables/feature_importance.csv")
    print("  outputs/figures/feature_importance.png")

    if dominance is not None:
        dominance.to_csv(OUT_TABLES / "channel_dominance.csv", index=False)
        plot_channel_dominance(dominance, importance,
                               OUT_FIGURES / "channel_dominance.png")
        print("  outputs/tables/channel_dominance.csv")
        print("  outputs/figures/channel_dominance.png")

    write_report(importance,
                 dominance if dominance is not None else pd.DataFrame(
                     columns=["feature", "r2_vs_level", "spearman_rho"]),
                 control, full_accuracy, chance,
                 OUT_REPORTS / "feature_analysis.md")
    print("  outputs/reports/feature_analysis.md")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
