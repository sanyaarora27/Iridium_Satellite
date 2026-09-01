"""
11_spoofing_attack.py
=====================

PURPOSE
-------
Step 13 of the 29 June task list: a feature-space spoofing experiment.

The task brief says to attempt the attack "only after the classifier works".
Two models were tested in this project:

  - 28 hand-crafted waveform features : 23.6%, p = 0.12 vs chance
                                        (does NOT beat chance)
  - receiver-side metadata            : 27.3%, p = 0.0002 vs chance
    (level, noise, centre frequency)    (DOES beat chance)

Both are attacked here. The waveform model is included because the task
asks for it and because attacking a chance-level classifier is itself
informative -- it establishes that no perturbation is required to defeat
it. The metadata model is the substantive target.

WHAT THE ATTACK DOES
--------------------
For an ordered pair of satellites (source S, target T), take genuine
messages from S and move their feature vectors toward T's class mean:

    x' = x + alpha * (mu_T - mu_S)

alpha = 0 leaves the message untouched; alpha = 1 places it exactly at the
target's mean. The attack succeeds when the classifier labels x' as T.

The reported quantity is the smallest alpha at which at least half of the
source messages are misclassified as the target. Small alpha means the
classes sit close together and impersonation is cheap.

WHY alpha IS EXPRESSED IN WITHIN-CLASS STANDARD DEVIATIONS
-----------------------------------------------------------
An alpha value alone is not meaningful: whether a shift is large depends on
how much the feature already varies naturally. The required shift per
feature is therefore also reported in units of the within-class standard
deviation. A shift of 0.2 sigma is smaller than the message-to-message
variation the receiver already sees and is undetectable in principle; a
shift of 5 sigma would be conspicuous.

ATTACKER CONTROLLABILITY
------------------------
A feature being easy to shift numerically is not the same as an attacker
being able to shift it in the field. Each input is therefore classified by
how directly an adversary governs it:

  direct   -- set by the transmitter (carrier frequency, transmit power,
              and any self-declared payload field)
  indirect -- follows from choices the attacker makes, such as position
  none     -- a property of the receiver or environment

USAGE
-----
    python scripts/11_spoofing_attack.py

OUTPUTS
-------
    outputs/tables/spoofing_results.csv
    outputs/tables/single_feature_attack.csv
    outputs/figures/spoofing_attack.png
    outputs/reports/spoofing_attack.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# --- PATHS ----------------------------------------------------------------
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
ALPHA_GRID    = np.arange(0.0, 2.01, 0.05)
SUCCESS_LEVEL = 0.50          # fraction of messages that must flip

NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}

# Receiver-side metadata only: the variant that beat chance at p = 0.0002.
# ra_alt is excluded because it is decoded from the message payload rather
# than measured by the receiver, so an attacker sets it by simply writing
# the value they want.
RX_METADATA = ["level", "noise", "center_frequency"]

CONTROLLABILITY = {
    "level":            ("indirect", "follows from transmit power and range"),
    "noise":            ("none",     "receiver noise floor, not attacker-set"),
    "center_frequency": ("direct",   "attacker chooses transmit frequency"),
    "ra_alt":           ("direct",   "self-declared field in the payload"),
}


# --- LOADING -------------------------------------------------------------
def load_metadata_column(column: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None


def load_all() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    features = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    key = next((c for c in ("global_index", "sample_id") if c in df.columns), None)
    rows = df[key].to_numpy(dtype=int)

    rx = []
    for col in RX_METADATA:
        full = load_metadata_column(col)
        if full is not None and rows.max() < len(full):
            df[f"meta_{col}"] = full[rows]
            rx.append(f"meta_{col}")
    return df, features, rx


def clean(X: np.ndarray) -> np.ndarray:
    """Convert non-finite values to NaN for train-fitted imputation."""
    X = X.astype(float).copy()
    X[~np.isfinite(X)] = np.nan
    return X


# --- ATTACK --------------------------------------------------------------
def run_attack(X: np.ndarray, y: np.ndarray,
               names: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    """
    Attack every ordered pair of satellites and record the perturbation
    required for impersonation.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_features=None if X.shape[1] <= 6 else "sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )),
    ])

    model.fit(X_tr, y_tr)

    # Use the imputer fitted ONLY on the training data.
    imputer = model.named_steps["imputer"]
    X_tr_imp = imputer.transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    baseline = float((model.predict(X_te) == y_te).mean())
    sats = np.unique(y)

    # Class statistics are calculated only from the imputed training set.
    means = {
        s: X_tr_imp[y_tr == s].mean(axis=0)
        for s in sats
    }

    spread = {
        s: X_tr_imp[y_tr == s].std(axis=0)
        for s in sats
    }

    rows = []
    curves = {}

    for src in sats:
        mask = (y_te == src)

        if mask.sum() < 10:
            continue

        X_src = X_te_imp[mask]

        for tgt in sats:
            if tgt == src:
                continue

            direction = means[tgt] - means[src]

            rates = []

            for a in ALPHA_GRID:
                X_attack = X_src + a * direction
                pred = model.predict(X_attack)
                rates.append(float((pred == tgt).mean()))

            rates = np.array(rates)
            curves[(int(src), int(tgt))] = rates

            hit = np.where(rates >= SUCCESS_LEVEL)[0]

            alpha_req = (
                float(ALPHA_GRID[hit[0]])
                if len(hit)
                else np.nan
            )

            if np.isfinite(alpha_req):
                pooled = np.sqrt(
                    (spread[src] ** 2 + spread[tgt] ** 2) / 2
                )

                shift_sigma = (
                    np.abs(alpha_req * direction) /
                    (pooled + 1e-12)
                )

                max_sigma = float(np.max(shift_sigma))
                mean_sigma = float(np.mean(shift_sigma))
                worst = names[int(np.argmax(shift_sigma))]

            else:
                max_sigma = np.nan
                mean_sigma = np.nan
                worst = "-"

            rows.append({
                "model": label,
                "source": int(src),
                "target": int(tgt),
                "alpha_required": alpha_req,
                "success_at_alpha0": rates[0],
                "success_at_alpha1": float(
                    rates[np.argmin(np.abs(ALPHA_GRID - 1.0))]
                ),
                "max_shift_sigma": max_sigma,
                "mean_shift_sigma": mean_sigma,
                "largest_shift_feature": worst,
            })

    return pd.DataFrame(rows), {
        "baseline": baseline,
        "curves": curves,
        "model": model,
        "X_te": X_te_imp,
        "y_te": y_te,
        "means": means,
        "spread": spread,
    }

    baseline = float((model.predict(X_te) == y_te).mean())
    sats = np.unique(y)

    # Class means from the TRAINING set only -- an attacker estimating a
    # target's profile would use observed traffic, not the evaluation set.
    means  = {s: X_tr[y_tr == s].mean(axis=0) for s in sats}
    spread = {s: X_tr[y_tr == s].std(axis=0)  for s in sats}

    rows = []
    curves = {}
    for src in sats:
        mask = (y_te == src)
        if mask.sum() < 10:
            continue
        X_src = X_te[mask]

        for tgt in sats:
            if tgt == src:
                continue
            direction = means[tgt] - means[src]

            rates = []
            for a in ALPHA_GRID:
                pred = model.predict(X_src + a * direction)
                rates.append(float((pred == tgt).mean()))
            rates = np.array(rates)
            curves[(int(src), int(tgt))] = rates

            hit = np.where(rates >= SUCCESS_LEVEL)[0]
            alpha_req = float(ALPHA_GRID[hit[0]]) if len(hit) else np.nan

            # Express the required move per feature in within-class sigmas
            if np.isfinite(alpha_req):
                pooled = np.sqrt((spread[src] ** 2 + spread[tgt] ** 2) / 2)
                shift_sigma = np.abs(alpha_req * direction) / (pooled + 1e-12)
                max_sigma  = float(np.max(shift_sigma))
                mean_sigma = float(np.mean(shift_sigma))
                worst = names[int(np.argmax(shift_sigma))]
            else:
                max_sigma = mean_sigma = np.nan
                worst = "-"

            rows.append({
                "model":            label,
                "source":           int(src),
                "target":           int(tgt),
                "alpha_required":   alpha_req,
                "success_at_alpha0": rates[0],
                "success_at_alpha1": float(rates[np.argmin(np.abs(ALPHA_GRID - 1.0))]),
                "max_shift_sigma":  max_sigma,
                "mean_shift_sigma": mean_sigma,
                "largest_shift_feature": worst,
            })

    return pd.DataFrame(rows), {"baseline": baseline, "curves": curves,
                                "model": model, "X_te": X_te, "y_te": y_te,
                                "means": means, "spread": spread}


def single_feature_attack(state: dict, names: list[str],
                          label: str) -> pd.DataFrame:
    """
    Perturb ONE feature at a time to answer the task's question directly:
    which features are easiest to manipulate?

    For each feature and each ordered pair, only that feature is moved to
    the target's mean value; everything else is left untouched. Features
    that produce many misclassifications on their own are the ones an
    attacker would target first.
    """
    model, X_te, y_te = state["model"], state["X_te"], state["y_te"]
    means = state["means"]
    sats = np.unique(y_te)

    rows = []
    for j, name in enumerate(names):
        flips = 0
        total = 0
        for src in sats:
            mask = (y_te == src)
            if mask.sum() < 10:
                continue
            for tgt in sats:
                if tgt == src:
                    continue
                X_mod = X_te[mask].copy()
                X_mod[:, j] = means[tgt][j]     # this feature only
                flips += int((model.predict(X_mod) == tgt).sum())
                total += int(mask.sum())

        base, why = CONTROLLABILITY.get(name.replace("meta_", ""),
                                        ("indirect", "derived from the waveform"))
        rows.append({
            "model":            label,
            "feature":          name.replace("meta_", ""),
            "impersonation_rate": flips / total if total else np.nan,
            "attacker_control": base,
            "note":             why,
        })

    return (pd.DataFrame(rows)
            .sort_values("impersonation_rate", ascending=False)
            .reset_index(drop=True))


# --- PLOT ----------------------------------------------------------------
def plot_attack(state_feat: dict, state_meta: dict | None, path: Path) -> None:
    n = 2 if state_meta else 1
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), squeeze=False)

    for ax, state, title in zip(
            axes[0],
            [state_feat] + ([state_meta] if state_meta else []),
            ["28 waveform features"] + (["receiver-side metadata"]
                                        if state_meta else [])):
        for (src, tgt), rates in state["curves"].items():
            ax.plot(ALPHA_GRID, rates, lw=1, alpha=0.6,
                    label=f"{src}->{tgt}")
        ax.axhline(SUCCESS_LEVEL, color="crimson", ls="--", lw=1.2,
                   label="50% success")
        ax.set_xlabel(r"Perturbation $\alpha$ toward target class mean")
        ax.set_ylabel("Fraction misclassified as target")
        ax.set_title(f"{title}\nbaseline accuracy {state['baseline']:.1%}",
                     fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
        if len(state["curves"]) <= 12:
            ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Spoofing attack in feature space (Step 13)")
    print("=" * 72)

    df, features, rx = load_all()
    y = df["satellite_id"].to_numpy()
    print(f"\n{len(df):,} messages, {len(features)} waveform features, "
          f"{len(rx)} receiver-side metadata columns")

    # ---- Attack 1: waveform features -----------------------------------
    print("\n" + "-" * 72)
    print("A. ATTACK ON THE 28 WAVEFORM FEATURES")
    print("-" * 72)
    X_feat = clean(df[features].to_numpy())
    res_feat, state_feat = run_attack(X_feat, y, features, "28 waveform features")
    print(f"  Baseline accuracy: {state_feat['baseline']:.4f}")
    print(f"  Impersonation succeeds with NO modification "
          f"(alpha = 0) in {res_feat['success_at_alpha0'].mean():.1%} of "
          f"attempts on average.")
    solved = res_feat["alpha_required"].notna().sum()
    print(f"  Pairs impersonated at some alpha <= 2: {solved} / {len(res_feat)}")
    if solved:
        print(f"  Median alpha required: "
              f"{res_feat['alpha_required'].median():.2f}")

    # ---- Attack 2: receiver-side metadata ------------------------------
    res_meta, state_meta = None, None
    if rx:
        print("\n" + "-" * 72)
        print("B. ATTACK ON THE RECEIVER-SIDE METADATA MODEL")
        print("-" * 72)
        X_meta = clean(df[rx].to_numpy())
        res_meta, state_meta = run_attack(X_meta, y, rx, "receiver-side metadata")
        print(f"  Baseline accuracy: {state_meta['baseline']:.4f}")
        solved_m = res_meta["alpha_required"].notna().sum()
        print(f"  Pairs impersonated at some alpha <= 2: "
              f"{solved_m} / {len(res_meta)}")
        if solved_m:
            print(f"  Median alpha required: "
                  f"{res_meta['alpha_required'].median():.2f}")
            print(f"  Median largest per-feature shift: "
                  f"{res_meta['max_shift_sigma'].median():.2f} sigma")
            print("\n  Worst pairs (smallest perturbation needed):")
            best = res_meta.dropna(subset=["alpha_required"]).nsmallest(
                5, "alpha_required")
            print(f"    {'src->tgt':<12}{'alpha':>8}{'max shift':>12}"
                  f"  {'feature moved most'}")
            for _, r in best.iterrows():
                print(f"    {int(r['source'])}->{int(r['target']):<8}"
                      f"{r['alpha_required']:>8.2f}"
                      f"{r['max_shift_sigma']:>11.2f}s"
                      f"  {r['largest_shift_feature'].replace('meta_','')}")

    # ---- Single-feature attack -----------------------------------------
    print("\n" + "-" * 72)
    print("C. WHICH FEATURES ARE EASIEST TO MANIPULATE?")
    print("-" * 72)
    single = single_feature_attack(state_feat, features, "28 waveform features")
    if state_meta:
        single = pd.concat([
            single,
            single_feature_attack(state_meta, rx, "receiver-side metadata")
        ], ignore_index=True)

    for label in single["model"].unique():
        sub = single[single["model"] == label].head(6)
        print(f"\n  {label}:")
        print(f"    {'feature':<22}{'impersonation':>15}{'control':>12}")
        for _, r in sub.iterrows():
            print(f"    {r['feature']:<22}{r['impersonation_rate']:>14.1%}"
                  f"{r['attacker_control']:>12}")

    # ---- Outputs --------------------------------------------------------
    print("\n" + "-" * 72)
    print("Writing outputs...")
    all_res = (pd.concat([res_feat, res_meta], ignore_index=True)
               if res_meta is not None else res_feat)
    all_res.to_csv(OUT_TABLES / "spoofing_results.csv", index=False)
    single.to_csv(OUT_TABLES / "single_feature_attack.csv", index=False)
    plot_attack(state_feat, state_meta, OUT_FIGURES / "spoofing_attack.png")

    md = f"""# Spoofing attack in feature space (Step 13)

## Design

For an ordered pair of satellites (source S, target T), genuine messages
from S are moved toward T's class mean in feature space:

    x' = x + alpha * (mu_T - mu_S)

alpha = 0 leaves the message unmodified; alpha = 1 places it exactly at the
target's mean. The attack succeeds when the classifier assigns x' to T. The
reported quantity is the smallest alpha at which at least
{SUCCESS_LEVEL:.0%} of source messages are misclassified as the target.

Class means are estimated from the training split only, reflecting an
adversary who profiles a target from observed traffic.

## A. Attack on the 28 waveform features

Baseline accuracy: {state_feat['baseline']:.1%} (chance is approximately 20.8%).

Because this classifier does not perform above chance
(McNemar p = 0.12), impersonation succeeds
**{res_feat['success_at_alpha0'].mean():.1%} of the time with no
modification whatsoever**. There is no meaningful attack to mount: the
classifier already assigns source messages to arbitrary classes at
approximately the base rate.

This is the correct result to report for a control that does not
discriminate. An attacker need not manipulate any waveform property,
because the defender cannot distinguish transmitters in the first place.

"""
    if res_meta is not None:
        cheap = res_meta.dropna(subset=["alpha_required"])
        md += f"""## B. Attack on the receiver-side metadata model

Baseline accuracy: {state_meta['baseline']:.1%}. This is the only model in
the project that classifies above chance (McNemar p = 0.0002), so it is the
only one for which the attack is meaningful.

| Source | Target | alpha required | Largest per-feature shift | Feature moved most |
|-------:|-------:|---------------:|--------------------------:|--------------------|
"""
        for _, r in cheap.nsmallest(10, "alpha_required").iterrows():
            md += (f"| {int(r['source'])} | {int(r['target'])} "
                   f"| {r['alpha_required']:.2f} "
                   f"| {r['max_shift_sigma']:.2f} sigma "
                   f"| `{r['largest_shift_feature'].replace('meta_','')}` |\n")

        md += f"""
Median alpha required across impersonated pairs:
{cheap['alpha_required'].median():.2f}. Median largest per-feature shift:
{cheap['max_shift_sigma'].median():.2f} within-class standard deviations.

Interpreting the shift in sigma units matters. A perturbation smaller than
one within-class standard deviation is smaller than the variation the
receiver already observes between genuine messages from the same
transmitter, and therefore cannot be flagged as anomalous without also
rejecting legitimate traffic.

"""

    md += """## C. Which inputs are easiest to manipulate?

Each input was perturbed on its own, with all others left untouched, and
the resulting impersonation rate recorded. Numerical ease is reported
alongside how directly an adversary governs the quantity in practice.

| Model | Input | Impersonation rate | Attacker control | Basis |
|-------|-------|-------------------:|------------------|-------|
"""
    for _, r in single.iterrows():
        md += (f"| {r['model']} | `{r['feature']}` "
               f"| {r['impersonation_rate']:.1%} "
               f"| {r['attacker_control']} | {r['note']} |\n")

    md += """
## D. Security interpretation

The two models fail in different ways, and the distinction matters for the
control assessment.

The waveform-feature model offers no protection because it does not
discriminate: impersonation succeeds without any manipulation. A control
that cannot separate legitimate transmitters cannot detect an illegitimate
one.

The receiver-side metadata model does discriminate, but what it
discriminates on is operating context -- carrier frequency, received power
and noise floor -- rather than transmitter hardware. Carrier frequency is
set directly by the transmitter. Received power follows from transmit power
and range, both of which an adversary chooses. Consequently an attacker who
transmits on the appropriate simplex channel at a plausible power from a
plausible location satisfies the model's expectations without imitating any
physical characteristic of the genuine satellite.

Neither model, therefore, provides transmitter authentication. The first
lacks discriminative power; the second discriminates on quantities under
the adversary's control.
"""
    (OUT_REPORTS / "spoofing_attack.md").write_text(md)

    print("  outputs/tables/spoofing_results.csv")
    print("  outputs/tables/single_feature_attack.csv")
    print("  outputs/figures/spoofing_attack.png")
    print("  outputs/reports/spoofing_attack.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
