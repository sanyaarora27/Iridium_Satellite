"""
10_beam_and_ablation.py
=======================

PURPOSE
-------
Two follow-up analyses demanded by the results of script 09.

  A. METADATA ABLATION
     McNemar showed that a model using four channel metadata values beats
     chance (p = 0.0099) while the 28 hand-crafted signal features do not
     (p = 0.1197). The variance decomposition then showed that `level`
     carries almost no satellite information (R^2 = 0.001 against satellite
     identity), so `level` cannot be what makes the metadata model work.

     This section trains the model on each metadata column alone, and on
     each column removed, to identify which one actually carries the
     signal. Naming the mechanism matters: if it is `center_frequency`,
     the model is keying on simplex channel assignment -- a protocol
     artifact, not a hardware fingerprint.

  B. WITHIN-BEAM CLASSIFICATION, ALL BEAMS
     Script 09 found 33.6% accuracy against 26.3% chance within beam 1 --
     the largest margin observed anywhere in this project. That was a
     single beam with n = 384, so it may be noise.

     This section repeats the analysis for every beam with enough messages,
     reports bootstrap confidence intervals, and aggregates across beams.
     Because many beams are tested, a Holm-Bonferroni correction is applied
     to the per-beam p-values; without it, testing twenty beams would be
     expected to produce one "significant" result by chance alone.

WHY BEAM MATTERS PHYSICALLY
---------------------------
Each Iridium satellite transmits through 48 spot beams with distinct
antenna patterns. The beam index identifies the position within that
pattern, so holding beam constant fixes the approximate angle from
boresight and therefore the antenna gain toward the receiver. This is a
tighter channel control than holding elevation constant, which fixes only
the path geometry and not the transmit-side gain.

USAGE
-----
    python scripts/10_beam_and_ablation.py

OUTPUTS
-------
    outputs/tables/metadata_ablation.csv
    outputs/tables/within_beam_results.csv
    outputs/figures/within_beam.png
    outputs/reports/beam_and_ablation.md
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
N_BOOTSTRAP   = 2000
MIN_BEAM_SIZE = 250

NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}
METADATA_COLUMNS = ["level", "noise", "ra_alt", "center_frequency"]

def load_metadata_column(column: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None

def load_all() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    features = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    key = next((c for c in ("global_index", "sample_id") if c in df.columns), None)
    rows = df[key].to_numpy(dtype=int)

    for col in METADATA_COLUMNS + ["ra_cell"]:
        full = load_metadata_column(col)
        if full is not None and rows.max() < len(full):
            df[f"meta_{col}"] = full[rows]
    return df, features

def clean_matrix(X: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the column median."""
    X = X.astype(float).copy()
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0
    return X

def evaluate(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED,
             max_features=None) -> dict:
    """
    Train a Random Forest and evaluate against the majority-class baseline
    on the same test split, returning paired correctness vectors so that
    McNemar's test can be applied.
    """
    if len(np.unique(y)) < 2 or len(y) < 50:
        return {}
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return {}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=seed, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        # max_features=None means every split considers all features.
        # The default of sqrt(n_features) is sensible for 28 features but
        # erratic for 1-4: with three inputs each split would sample only
        # one or two, so an informative column is frequently unavailable
        # and subset comparisons become dominated by that sampling noise.
        ("clf",   RandomForestClassifier(n_estimators=200,
                                         max_features=max_features,
                                         random_state=seed, n_jobs=-1)),
    ]).fit(X_tr, y_tr)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)

    correct_model = (model.predict(X_te) == y_te)
    correct_chance = (dummy.predict(X_te) == y_te)

    # Bootstrap CI on the model's accuracy
    rng = np.random.default_rng(seed)
    m = len(correct_model)
    boot = [correct_model[rng.integers(0, m, m)].mean() for _ in range(N_BOOTSTRAP)]

    # Exact McNemar against the chance baseline
    n01 = int(np.sum(~correct_model & correct_chance))
    n10 = int(np.sum(correct_model & ~correct_chance))
    n_disc = n01 + n10
    p = (float(scipy_stats.binomtest(n10, n_disc, 0.5).pvalue)
         if n_disc > 0 else 1.0)

    return {
        "accuracy": float(correct_model.mean()),
        "chance":   float(correct_chance.mean()),
        "ci_low":   float(np.percentile(boot, 2.5)),
        "ci_high":  float(np.percentile(boot, 97.5)),
        "p_value":  p,
        "n_test":   int(m),
        "n_discordant": n_disc,
    }

def metadata_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which metadata column carries the class-discriminative signal.

    Two complementary views:
      "only X"      -- how much can X alone achieve?
      "without X"   -- how much is lost when X is removed?

    A column that scores well alone AND whose removal causes a large drop
    is the one doing the work.
    """
    cols = [f"meta_{c}" for c in METADATA_COLUMNS if f"meta_{c}" in df.columns]
    y = df["satellite_id"].to_numpy()
    rows = []

    # Provenance matters more than performance here.
    #
    #   receiver-side : measured by the ground station from the waveform
    #                   (level, noise, centre frequency). An attacker
    #                   controls these only indirectly, through transmit
    #                   power and position.
    #
    #   payload       : decoded from the message body. Iridium Ring Alert
    #                   messages carry the satellite's own reported
    #                   position, so ra_lat / ra_lon / ra_alt are
    #                   self-declared. Identifying a transmitter from a
    #                   field it broadcasts about itself is not
    #                   fingerprinting; it is reading the label, and it
    #                   offers no resistance to spoofing.
    provenance = {"level": "receiver-side", "noise": "receiver-side",
                  "center_frequency": "receiver-side", "ra_alt": "payload",
                  "ra_lat": "payload", "ra_lon": "payload"}

    full = evaluate(clean_matrix(df[cols].to_numpy()), y)
    if full:
        rows.append({"variant": f"all {len(cols)} columns",
                     "columns": ", ".join(c.replace("meta_", "") for c in cols),
                     "provenance": "mixed", **full})

    for c in cols:
        name = c.replace("meta_", "")
        r = evaluate(clean_matrix(df[[c]].to_numpy()), y)
        if r:
            rows.append({"variant": f"only {name}", "columns": name,
                         "provenance": provenance.get(name, "?"), **r})

    if len(cols) > 1:
        for c in cols:
            name = c.replace("meta_", "")
            rest = [x for x in cols if x != c]
            r = evaluate(clean_matrix(df[rest].to_numpy()), y)
            if r:
                rows.append({"variant": f"without {name}",
                             "columns": ", ".join(x.replace("meta_", "")
                                                  for x in rest),
                             "provenance": "mixed", **r})

    # Receiver-side columns only -- the model an attacker cannot trivially
    # forge, and therefore the only variant with authentication relevance.
    rx = [c for c in cols if provenance.get(c.replace("meta_", "")) == "receiver-side"]
    if rx and len(rx) < len(cols):
        r = evaluate(clean_matrix(df[rx].to_numpy()), y)
        if r:
            rows.append({"variant": "receiver-side only",
                         "columns": ", ".join(c.replace("meta_", "") for c in rx),
                         "provenance": "receiver-side", **r})

    pay = [c for c in cols if provenance.get(c.replace("meta_", "")) == "payload"]
    if pay and len(pay) < len(cols):
        r = evaluate(clean_matrix(df[pay].to_numpy()), y)
        if r:
            rows.append({"variant": "payload only",
                         "columns": ", ".join(c.replace("meta_", "") for c in pay),
                         "provenance": "payload", **r})

    return pd.DataFrame(rows)

def holm_bonferroni(p_values: list[float]) -> list[bool]:
    """
    Holm-Bonferroni correction for multiple comparisons.

    Testing many beams independently at alpha = 0.05 would be expected to
    produce roughly one false positive per twenty beams. Holm's step-down
    procedure controls the family-wise error rate while being less
    conservative than plain Bonferroni: p-values are sorted ascending and
    compared against alpha/(m - i), stopping at the first failure.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= 0.05 / (m - rank):
            reject[idx] = True
        else:
            break
    return reject

def within_beam_analysis(df: pd.DataFrame,
                         features: list[str]) -> pd.DataFrame:
    """Classification within each beam that has enough messages."""
    if "meta_ra_cell" not in df.columns:
        return pd.DataFrame()

    X = df[features].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()
    beams = df["meta_ra_cell"].to_numpy()

    rows = []
    for beam, count in pd.Series(beams).value_counts().items():
        if count < MIN_BEAM_SIZE:
            continue
        mask = (beams == beam)
        r = evaluate(X[mask], y[mask], max_features="sqrt")
        if not r:
            continue
        rows.append({"beam": beam, "n_messages": int(count),
                     "n_classes": int(len(np.unique(y[mask]))),
                     "margin": r["accuracy"] - r["chance"], **r})

    table = pd.DataFrame(rows)
    if len(table):
        table["significant_holm"] = holm_bonferroni(table["p_value"].tolist())
        table = table.sort_values("margin", ascending=False).reset_index(drop=True)
    return table

def plot_beams(table: pd.DataFrame, path: Path) -> None:
    if not len(table):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    t = table.sort_values("beam")
    x = np.arange(len(t))
    ax1.bar(x - 0.2, t["accuracy"], 0.4, label="Random Forest",
            color="steelblue",
            yerr=[t["accuracy"] - t["ci_low"], t["ci_high"] - t["accuracy"]],
            capsize=3)
    ax1.bar(x + 0.2, t["chance"], 0.4, label="Chance", color="grey")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in t["beam"]], fontsize=8)
    ax1.set_xlabel("Beam (ra_cell)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Classification within each beam", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    colours = ["seagreen" if s else "lightsteelblue"
               for s in t["significant_holm"]]
    ax2.bar(x, t["margin"], color=colours)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in t["beam"]], fontsize=8)
    ax2.set_xlabel("Beam (ra_cell)")
    ax2.set_ylabel("Accuracy minus chance")
    ax2.set_title("Margin over chance\n"
                  "(green = significant after Holm-Bonferroni)",
                  fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    print("=" * 72)
    print("Metadata ablation and within-beam analysis")
    print("=" * 72)

    df, features = load_all()
    print(f"\n{len(df):,} messages, {len(features)} features")

    print("\n" + "-" * 72)
    print("A. METADATA ABLATION - which column carries the signal?")
    print("-" * 72)
    ablation = metadata_ablation(df)
    if len(ablation):
        print(f"\n  {'variant':<26}{'provenance':<16}{'acc':>8}"
              f"{'chance':>9}{'p':>10}")
        print("  " + "-" * 67)
        for _, r in ablation.iterrows():
            star = " *" if r["p_value"] < 0.05 else ""
            print(f"  {r['variant']:<26}{str(r.get('provenance','')):<16}"
                  f"{r['accuracy']:>8.4f}{r['chance']:>9.4f}"
                  f"{r['p_value']:>10.4f}{star}")
        print("\n  * = significantly better than chance (p < 0.05)")

        singles = ablation[ablation["variant"].str.startswith("only")]
        if len(singles):
            best = singles.loc[singles["accuracy"].idxmax()]
            print(f"\n  Strongest single column: {best['columns']} "
                  f"({best['accuracy']:.4f}, p = {best['p_value']:.4f})")

    print("\n" + "-" * 72)
    print("B. WITHIN-BEAM CLASSIFICATION (all beams)")
    print("-" * 72)
    beams = within_beam_analysis(df, features)

    if not len(beams):
        print(f"  No beam has at least {MIN_BEAM_SIZE} messages.")
    else:
        print(f"\n  {len(beams)} beams with >= {MIN_BEAM_SIZE} messages\n")
        print(f"  {'beam':>6}{'n':>7}{'acc':>9}{'chance':>9}{'margin':>9}"
              f"{'p':>10}{'Holm':>7}")
        print("  " + "-" * 57)
        for _, r in beams.iterrows():
            print(f"  {int(r['beam']):>6}{int(r['n_messages']):>7,}"
                  f"{r['accuracy']:>9.4f}{r['chance']:>9.4f}"
                  f"{r['margin']:>+9.4f}{r['p_value']:>10.4f}"
                  f"{'  yes' if r['significant_holm'] else '   no':>7}")

        n_sig = int(beams["significant_holm"].sum())
        mean_margin = float(beams["margin"].mean())
        print(f"\n  Mean margin over chance: {mean_margin:+.4f}")
        print(f"  Beams significant after Holm-Bonferroni: {n_sig} / {len(beams)}")

        # Sign test: are margins consistently positive across beams?
        pos = int((beams["margin"] > 0).sum())
        sign_p = float(scipy_stats.binomtest(pos, len(beams), 0.5).pvalue)
        print(f"  Beams with positive margin: {pos} / {len(beams)}  "
              f"(sign test p = {sign_p:.4f})")

        if n_sig > 0:
            print("\n  => At least one beam shows classification above chance")
            print("     after correction for multiple comparisons. Holding")
            print("     beam constant may expose transmitter-specific")
            print("     structure that beam variation otherwise masks.")
        elif sign_p < 0.05:
            print("\n  => No individual beam survives correction, but margins")
            print("     are consistently positive across beams, which is")
            print("     weak evidence of a small real effect.")
        else:
            print("\n  => No evidence that holding beam constant recovers")
            print("     transmitter information. The single-beam result in")
            print("     script 09 does not replicate across beams.")

    print("\n" + "-" * 72)
    print("Writing outputs...")
    if len(ablation):
        ablation.to_csv(OUT_TABLES / "metadata_ablation.csv", index=False)
        print("  outputs/tables/metadata_ablation.csv")
    if len(beams):
        beams.to_csv(OUT_TABLES / "within_beam_results.csv", index=False)
        plot_beams(beams, OUT_FIGURES / "within_beam.png")
        print("  outputs/tables/within_beam_results.csv")
        print("  outputs/figures/within_beam.png")

    md = "# Metadata ablation and within-beam analysis\n\n"
    md += """## A. Which metadata column carries the signal?

McNemar's test showed that a Random Forest using four channel metadata
values classifies above chance, while the same model using 28 hand-crafted
signal features does not. The variance decomposition then showed that
`level` explains almost none of the variation associated with satellite
identity, so `level` cannot be responsible. This ablation isolates the
column that is.

| Variant | Accuracy | Chance | p (McNemar) |
|---------|---------:|-------:|------------:|
"""
    for _, r in ablation.iterrows():
        md += (f"| {r['variant']} | {r['accuracy']:.4f} | {r['chance']:.4f} "
               f"| {r['p_value']:.4f} |\n")

    md += f"""

## B. Within-beam classification

Each Iridium satellite transmits through 48 spot beams with distinct
antenna patterns. Holding the beam index constant fixes the approximate
angle from boresight, and therefore the transmit antenna gain toward the
receiver -- a tighter channel control than fixing elevation, which
constrains only the propagation path.

Beams with at least {MIN_BEAM_SIZE} messages were analysed separately.
Because several beams are tested, per-beam p-values are corrected using the
Holm-Bonferroni step-down procedure; without correction, testing twenty
beams at alpha = 0.05 would be expected to yield one spurious result.

"""
    if len(beams):
        md += "| Beam | Messages | Accuracy | 95% CI | Chance | Margin | p | Significant (Holm) |\n"
        md += "|---|---:|---:|:---:|---:|---:|---:|---|\n"
        for _, r in beams.iterrows():
            md += (f"| {int(r['beam'])} | {int(r['n_messages']):,} "
                   f"| {r['accuracy']:.4f} "
                   f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                   f"| {r['chance']:.4f} | {r['margin']:+.4f} "
                   f"| {r['p_value']:.4f} "
                   f"| {'yes' if r['significant_holm'] else 'no'} |\n")
        md += (f"\nMean margin over chance across beams: "
               f"{beams['margin'].mean():+.4f}. "
               f"{int(beams['significant_holm'].sum())} of {len(beams)} beams "
               f"remain significant after correction.\n")
    else:
        md += f"No beam contained at least {MIN_BEAM_SIZE} messages.\n"

    (OUT_REPORTS / "beam_and_ablation.md").write_text(md)
    print("  outputs/reports/beam_and_ablation.md")
    print("=" * 72)

if __name__ == "__main__":
    main()
