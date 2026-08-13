"""
========================

PURPOSE
-------
Three evaluations drawn from the 2026 satellite-security literature that
the existing pipeline does not perform.

  A. FEATURE SET COMPARISON INCLUDING v3
     Does adding power-amplifier nonlinearity features change the result?
     A March 2026 theoretical study reports that IQ imbalance may carry
     insufficient identifying information under some modulations while
     amplifier nonlinearity is more reliable. v2 tested the former and did
     not improve on v1; v3 tests the latter.

  B. CROSS-DOMAIN GENERALISATION
     Current channel-robustness work does not report a single accuracy. It
     reports accuracy WITHIN a channel condition and ACROSS conditions, and
     treats the gap as the result. CrossRF, for example, reports 26.39% for
     conventional methods across channels against 99.03% with domain
     adaptation.

     Every evaluation in this project so far has been within-domain: a
     random split mixes all beams and all passes across train and test.
     Two domain variables are available. Beam (`ra_cell`) fixes the
     transmit antenna pattern; pass fixes geometry and time. Training on
     one set of domains and testing on a disjoint set measures whether the
     features describe the transmitter or the conditions of observation.

  C. OPEN-SET REJECTION
     Operational satellite monitoring requires detecting unknown or
     anomalous transmitters, not assigning every signal to one of a fixed
     set. Current guidance lists a rejection mechanism as a required
     component of physical-layer authentication.

     Every model built here is closed-set: it must output one of five
     labels and cannot answer "none of these". This section holds out one
     satellite entirely, trains on the remaining four, and measures whether
     the held-out transmitter can be detected as unknown from classifier
     confidence -- reported as AUROC over held-out choices.

USAGE
-----
    python scripts/21_openset_and_domain.py

Requires scripts 03, 13 and 16 to have been run.

OUTPUTS
-------
    outputs/tables/feature_set_comparison.csv
    outputs/tables/cross_domain_results.csv
    outputs/tables/openset_results.csv
    outputs/figures/openset_and_domain.png
    outputs/reports/openset_and_domain.md
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
NON_FEATURE   = {"sample_id", "global_index", "index",
                 "Unnamed: 0", "satellite_id"}
PASS_GAP_S    = 20 * 60


def load_col(name):
    files = sorted(DATA_RAW.glob(f"{name}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None


def load_all() -> tuple[pd.DataFrame, dict]:
    """Merge v1, v2 and v3 feature tables and attach domain variables."""
    v1 = pd.read_csv(OUT_TABLES / "features.csv")
    sets = {"v1 (hand-crafted)": [c for c in v1.columns if c not in NON_FEATURE]}
    df = v1

    for path, label in [("features_v2.csv", "v2 (amplitude-invariant)"),
                        ("features_v3.csv", "v3 (PA nonlinearity)")]:
        p = OUT_TABLES / path
        if not p.exists():
            print(f"  {path} not found -- skipping {label}")
            continue
        extra = pd.read_csv(p)
        sets[label] = [c for c in extra.columns if c not in NON_FEATURE]
        df = df.merge(extra.drop(columns=["satellite_id"]),
                      on="global_index", how="inner")

    # Domain variables
    rows = df["global_index"].to_numpy(dtype=int)
    cell = load_col("ra_cell")
    if cell is not None and rows.max() < len(cell):
        df["beam"] = cell[rows]

    ts = load_col("timestamp_global")
    if ts is not None and rows.max() < len(ts):
        t = ts[rows].astype(float)
        if np.nanmedian(np.abs(t)) > 1e17:
            t = t / 1e9
        df["time_s"] = t
        # Passes: contiguous runs per satellite separated by long gaps
        pass_id = np.empty(len(df), dtype=int)
        nxt = 0
        sat = df["satellite_id"].to_numpy()
        for s in np.unique(sat):
            idx = np.where(sat == s)[0]
            order = idx[np.argsort(t[idx])]
            cur = nxt
            pass_id[order[0]] = cur
            for a, b in zip(order[:-1], order[1:]):
                if t[b] - t[a] > PASS_GAP_S:
                    cur += 1
                pass_id[b] = cur
            nxt = cur + 1
        df["pass_id"] = pass_id

    return df, sets


def clean(X):
    X = X.astype(float).copy()
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0
    return X


def forest():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=300,
                                         random_state=RANDOM_SEED, n_jobs=-1)),
    ])


def mcnemar(a, b):
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(scipy_stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def bootstrap_ci(correct):
    rng = np.random.default_rng(RANDOM_SEED)
    m = len(correct)
    s = [correct[rng.integers(0, m, m)].mean() for _ in range(2000)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Feature sets, cross-domain generalisation, and open-set rejection")
    print("=" * 72)

    print("\nLoading...")
    df, sets = load_all()
    y = df["satellite_id"].to_numpy()
    print(f"  {len(df):,} messages")
    for k, v in sets.items():
        print(f"    {k:<28}{len(v):>3} features")

    # ---- A. Feature set comparison --------------------------------------
    print("\n" + "-" * 72)
    print("A. FEATURE SET COMPARISON")
    print("-" * 72)

    idx = np.arange(len(y))
    itr, ite, ytr, yte = train_test_split(
        idx, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    dummy = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(itr), 1)), ytr)
    chance_correct = (dummy.predict(np.zeros((len(ite), 1))) == yte)

    comparison = []
    all_features = []
    correct_by_set = {}
    for label, cols in sets.items():
        all_features += cols
        X = clean(df[cols].to_numpy())
        m = forest().fit(X[itr], ytr)
        correct = (m.predict(X[ite]) == yte)
        correct_by_set[label] = correct
        lo, hi = bootstrap_ci(correct)
        comparison.append({
            "feature_set": label, "n_features": len(cols),
            "accuracy": float(correct.mean()), "ci_low": lo, "ci_high": hi,
            "p_vs_chance": mcnemar(chance_correct, correct)})

    # All sets combined
    if len(sets) > 1:
        X = clean(df[all_features].to_numpy())
        m = forest().fit(X[itr], ytr)
        correct = (m.predict(X[ite]) == yte)
        correct_by_set["all combined"] = correct
        lo, hi = bootstrap_ci(correct)
        comparison.append({
            "feature_set": "all combined", "n_features": len(all_features),
            "accuracy": float(correct.mean()), "ci_low": lo, "ci_high": hi,
            "p_vs_chance": mcnemar(chance_correct, correct)})

    lo, hi = bootstrap_ci(chance_correct)
    comparison.insert(0, {"feature_set": "chance", "n_features": 0,
                          "accuracy": float(chance_correct.mean()),
                          "ci_low": lo, "ci_high": hi, "p_vs_chance": 1.0})

    comp = pd.DataFrame(comparison)
    print(f"\n  {'feature set':<28}{'n':>5}{'acc':>9}{'95% CI':>20}{'p':>10}")
    print("  " + "-" * 72)
    for _, r in comp.iterrows():
        star = " *" if r["p_vs_chance"] < 0.05 and r["feature_set"] != "chance" else ""
        print(f"  {r['feature_set']:<28}{int(r['n_features']):>5}"
              f"{r['accuracy']:>9.4f}   [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
              f"{r['p_vs_chance']:>10.4f}{star}")

    if "v3 (PA nonlinearity)" in correct_by_set and "v1 (hand-crafted)" in correct_by_set:
        p = mcnemar(correct_by_set["v1 (hand-crafted)"],
                    correct_by_set["v3 (PA nonlinearity)"])
        print(f"\n  v3 vs v1 directly: p = {p:.4f}  "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    # ---- B. Cross-domain generalisation ---------------------------------
    print("\n" + "-" * 72)
    print("B. CROSS-DOMAIN GENERALISATION")
    print("-" * 72)
    print("\n  Within-domain: random split, all domains in train and test.")
    print("  Cross-domain:  train on one set of domains, test on a disjoint set.")
    print("  A large gap means the features describe conditions, not transmitters.\n")

    cross_rows = []
    best_label = max((c for c in comparison if c["feature_set"] != "chance"),
                     key=lambda c: c["accuracy"])["feature_set"]
    best_cols = (all_features if best_label == "all combined"
                 else sets[best_label])
    X_best = clean(df[best_cols].to_numpy())

    for domain in ("beam", "pass_id"):
        if domain not in df.columns:
            continue
        groups = df[domain].to_numpy()
        uniq = np.unique(groups)
        if len(uniq) < 4:
            print(f"  {domain}: only {len(uniq)} domains -- skipping")
            continue

        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = rng.permutation(uniq)
        half = len(shuffled) // 2
        set_a, set_b = set(shuffled[:half]), set(shuffled[half:])
        in_a = np.array([g in set_a for g in groups])

        # Both halves must contain every class for the comparison to mean
        # anything.
        if len(np.unique(y[in_a])) < len(np.unique(y)) or \
           len(np.unique(y[~in_a])) < len(np.unique(y)):
            print(f"  {domain}: split does not preserve all classes -- skipping")
            continue

        m = forest().fit(X_best[in_a], y[in_a])
        cross_acc = accuracy_score(y[~in_a], m.predict(X_best[~in_a]))
        d = DummyClassifier(strategy="most_frequent").fit(X_best[in_a], y[in_a])
        cross_chance = accuracy_score(y[~in_a], d.predict(X_best[~in_a]))

        # Within-domain reference on the same data volume
        i2 = np.arange(len(y))
        a2, b2, ya2, yb2 = train_test_split(
            i2, y, test_size=0.5, random_state=RANDOM_SEED, stratify=y)
        m2 = forest().fit(X_best[a2], ya2)
        within_acc = accuracy_score(yb2, m2.predict(X_best[b2]))
        d2 = DummyClassifier(strategy="most_frequent").fit(X_best[a2], ya2)
        within_chance = accuracy_score(yb2, d2.predict(X_best[b2]))

        cross_rows.append({
            "domain": domain, "n_domains": int(len(uniq)),
            "within_accuracy": within_acc, "within_chance": within_chance,
            "cross_accuracy": cross_acc, "cross_chance": cross_chance,
            "within_margin": within_acc - within_chance,
            "cross_margin": cross_acc - cross_chance})

        print(f"  {domain} ({len(uniq)} domains)")
        print(f"    within-domain: {within_acc:.4f}  "
              f"(chance {within_chance:.4f}, margin {within_acc-within_chance:+.4f})")
        print(f"    cross-domain:  {cross_acc:.4f}  "
              f"(chance {cross_chance:.4f}, margin {cross_acc-cross_chance:+.4f})")

    cross = pd.DataFrame(cross_rows)
    if len(cross):
        drop = float((cross["within_margin"] - cross["cross_margin"]).mean())
        print(f"\n  Mean margin lost when crossing domains: {drop:+.4f}")
        if drop > 0.02:
            print("  => Performance depends on the observation conditions, not")
            print("     only on the transmitter. This is the failure mode that")
            print("     motivates the channel-robust and domain-adaptation")
            print("     literature.")
        else:
            print("  => Little domain dependence, though with margins this")
            print("     small the comparison has limited power.")

    # ---- C. Open-set rejection ------------------------------------------
    print("\n" + "-" * 72)
    print("C. OPEN-SET REJECTION")
    print("-" * 72)
    print("\n  Each satellite is held out in turn. A model is trained on the")
    print("  remaining four, and its maximum class probability is used as a")
    print("  confidence score. AUROC measures whether the unseen transmitter")
    print("  receives systematically lower confidence than known ones.\n")

    open_rows = []
    sats = np.unique(y)
    for held in sats:
        known = (y != held)
        Xk, yk = X_best[known], y[known]
        if len(np.unique(yk)) < 2:
            continue

        itr2, ite2, ytr2, yte2 = train_test_split(
            np.arange(len(yk)), yk, test_size=TEST_FRACTION,
            random_state=RANDOM_SEED, stratify=yk)
        m = forest().fit(Xk[itr2], ytr2)

        conf_known   = m.predict_proba(Xk[ite2]).max(axis=1)
        conf_unknown = m.predict_proba(X_best[~known]).max(axis=1)

        labels = np.concatenate([np.zeros(len(conf_known)),
                                 np.ones(len(conf_unknown))])
        scores = np.concatenate([-conf_known, -conf_unknown])  # low conf = unknown
        auroc = float(roc_auc_score(labels, scores))

        open_rows.append({
            "held_out": int(held), "n_unknown": int((~known).sum()),
            "auroc": auroc,
            "mean_conf_known": float(conf_known.mean()),
            "mean_conf_unknown": float(conf_unknown.mean())})
        print(f"    hold out Sat {int(held):>3d}: AUROC = {auroc:.4f}   "
              f"conf known {conf_known.mean():.3f} vs "
              f"unknown {conf_unknown.mean():.3f}")

    openset = pd.DataFrame(open_rows)
    if len(openset):
        mean_auroc = float(openset["auroc"].mean())
        print(f"\n  Mean AUROC: {mean_auroc:.4f}   (0.5 = no ability to reject)")
        if mean_auroc > 0.70:
            print("  => Unknown transmitters receive measurably lower confidence.")
        elif mean_auroc > 0.55:
            print("  => Weak but non-trivial rejection ability.")
        else:
            print("  => Cannot distinguish unknown transmitters from known ones.")
            print("     A deployed system would admit any unenrolled signal.")

    # ---- Outputs --------------------------------------------------------
    print("\n" + "-" * 72)
    print("Writing outputs...")
    comp.to_csv(OUT_TABLES / "feature_set_comparison.csv", index=False)
    if len(cross):
        cross.to_csv(OUT_TABLES / "cross_domain_results.csv", index=False)
    if len(openset):
        openset.to_csv(OUT_TABLES / "openset_results.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    c = comp
    axes[0].bar(range(len(c)), c["accuracy"],
                yerr=[c["accuracy"] - c["ci_low"], c["ci_high"] - c["accuracy"]],
                capsize=4,
                color=["grey"] + ["steelblue"] * (len(c) - 1))
    axes[0].axhline(c.iloc[0]["accuracy"], color="crimson", ls="--", lw=1.2)
    axes[0].set_xticks(range(len(c)))
    axes[0].set_xticklabels([l.split(" (")[0] for l in c["feature_set"]],
                            rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Feature sets", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    if len(cross):
        x = np.arange(len(cross))
        axes[1].bar(x - 0.2, cross["within_margin"], 0.4,
                    label="within domain", color="seagreen")
        axes[1].bar(x + 0.2, cross["cross_margin"], 0.4,
                    label="across domains", color="darkorange")
        axes[1].axhline(0, color="black", lw=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cross["domain"])
        axes[1].set_ylabel("Accuracy above chance")
        axes[1].set_title("Cross-domain generalisation", fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    if len(openset):
        axes[2].bar([str(int(s)) for s in openset["held_out"]],
                    openset["auroc"], color="steelblue")
        axes[2].axhline(0.5, color="crimson", ls="--", lw=1.2,
                        label="no rejection ability")
        axes[2].set_xlabel("Held-out satellite")
        axes[2].set_ylabel("AUROC")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Open-set rejection", fontweight="bold")
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "openset_and_domain.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    md = """# Feature sets, cross-domain generalisation, and open-set rejection

Three evaluations drawn from the 2026 satellite-security literature.

## A. Feature set comparison

A March 2026 theoretical study of satellite RF fingerprint limits reports
that IQ imbalance may carry insufficient identifying information under some
modulation schemes, while power-amplifier nonlinearities are more reliable.
The v2 set tested the former and did not improve on v1; v3 adds AM/PM
conversion, envelope compression and spectral-regrowth features to test the
latter.

| Feature set | Features | Accuracy | 95% CI | p vs chance |
|-------------|---------:|---------:|:------:|------------:|
"""
    for _, r in comp.iterrows():
        md += (f"| {r['feature_set']} | {int(r['n_features'])} "
               f"| {r['accuracy']:.4f} "
               f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
               f"| {r['p_vs_chance']:.4f} |\n")

    md += """
## B. Cross-domain generalisation

Channel-robustness research reports accuracy within a channel condition and
across conditions separately, treating the gap as the result. All previous
evaluations in this project were within-domain: a random split places every
beam and every pass on both sides. Beam fixes the transmit antenna pattern;
pass fixes geometry and time.

| Domain | Domains | Within margin | Cross margin | Loss |
|--------|--------:|--------------:|-------------:|-----:|
"""
    for _, r in cross.iterrows():
        md += (f"| {r['domain']} | {int(r['n_domains'])} "
               f"| {r['within_margin']:+.4f} | {r['cross_margin']:+.4f} "
               f"| {r['within_margin'] - r['cross_margin']:+.4f} |\n")

    md += """
## C. Open-set rejection

Operational monitoring requires detecting unknown transmitters rather than
assigning every signal to a fixed set, and current guidance lists a
rejection mechanism as a required component of physical-layer
authentication. Every model built in this project is closed-set.

Each satellite is held out in turn, a model trained on the remaining four,
and maximum class probability used as a confidence score.

| Held out | Unknown messages | AUROC | Mean confidence (known) | Mean confidence (unknown) |
|---------:|-----------------:|------:|------------------------:|--------------------------:|
"""
    for _, r in openset.iterrows():
        md += (f"| {int(r['held_out'])} | {int(r['n_unknown']):,} "
               f"| {r['auroc']:.4f} | {r['mean_conf_known']:.3f} "
               f"| {r['mean_conf_unknown']:.3f} |\n")

    if len(openset):
        md += f"""
Mean AUROC {openset['auroc'].mean():.4f}, where 0.5 indicates no ability to
distinguish an unenrolled transmitter from an enrolled one.
"""

    (OUT_REPORTS / "openset_and_domain.md").write_text(md)
    print("  outputs/tables/feature_set_comparison.csv")
    print("  outputs/tables/cross_domain_results.csv")
    print("  outputs/tables/openset_results.csv")
    print("  outputs/figures/openset_and_domain.png")
    print("  outputs/reports/openset_and_domain.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
