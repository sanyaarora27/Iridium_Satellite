"""
============================

PURPOSE
-------
Restate the classification results as authentication error rates, which is
how the control would actually be assessed if deployed.

WHY THIS REFRAMING IS NECESSARY
-------------------------------
Accuracy answers "which satellite is this?". Authentication asks a
different question, and it is the one the supervisor's brief poses:

    "If a signal claims to be satellite 1 but the RF fingerprint classifier
     does not classify it as satellite 1, then there may be spoofing."

That is a verification decision, and it has two failure modes:

    False Reject Rate (FRR)  genuine traffic from satellite T is not
                             recognised as T, and legitimate messages are
                             flagged as spoofed
    False Accept Rate (FAR)  a message from a different transmitter is
                             accepted as T, and an impersonation succeeds

A control can only be judged against both simultaneously. Reporting
accuracy alone conceals that a 27% classifier rejects nearly three
quarters of authentic traffic.

WHAT THIS SCRIPT DOES
---------------------
  A. Decision-rule metrics. Under the argmax rule (accept the highest-
     scoring class), compute per-satellite FRR and FAR.

  B. Threshold sweep and EER. Random Forests emit class probabilities, so
     a confidence threshold can be applied: accept the claimed identity
     only if its probability exceeds tau. Sweeping tau traces the
     FAR/FRR trade-off and yields the Equal Error Rate, the point where
     the two are equal. EER is the standard single-number summary in the
     biometric and RF-fingerprinting literature, and is what SatIQ reports.

  C. Comparison against SatIQ's published figures on the same
     constellation.

USAGE
-----
    python scripts/12_authentication_metrics.py

OUTPUTS
-------
    outputs/tables/authentication_metrics.csv
    outputs/figures/authentication_tradeoff.png
    outputs/reports/authentication_metrics.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                    "Unnamed: 0", "satellite_id"}
RX_METADATA = ["level", "noise", "center_frequency"]

# Published figures for SatIQ (Smailes et al., 2025) on Iridium.
SATIQ_EER = 0.072
SATIQ_AUC = 0.96


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


# --- A. ARGMAX DECISION RULE ---------------------------------------------
def argmax_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Per-satellite FRR and FAR under the argmax rule.

    For satellite T:
        FRR(T) = fraction of genuine T messages NOT labelled T
                -> legitimate traffic that would be flagged as spoofed
        FAR(T) = fraction of non-T messages labelled T
-> impersonations of T that would be accepted
    """
    rows = []
    for sat in np.unique(y_true):
        genuine = (y_true == sat)
        others  = ~genuine
        frr = float((y_pred[genuine] != sat).mean())
        far = float((y_pred[others] == sat).mean()) if others.any() else np.nan
        rows.append({"satellite": int(sat),
                "n_genuine": int(genuine.sum()),
                "FRR": frr, "FAR": far})
    return pd.DataFrame(rows)


# --- B. THRESHOLD SWEEP AND EER ------------------------------------------
def threshold_sweep(y_true: np.ndarray,
                    proba: np.ndarray,
                    classes: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    """
    Sweep a confidence threshold over the verification decision.

    The verifier accepts a claimed identity T only when the model's
    probability for T exceeds tau. Raising tau makes acceptance stricter:
    FAR falls and FRR rises. The Equal Error Rate is the value where the
    two curves cross, and is the conventional single-number summary.

    Genuine trials  : every message paired with its true satellite
    Impostor trials : every message paired with each other satellite
    """
    genuine_scores, impostor_scores = [], []
    for i, true_label in enumerate(y_true):
        for j, cls in enumerate(classes):
            (genuine_scores if cls == true_label else impostor_scores
).append(proba[i, j])

    genuine  = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    taus = np.linspace(0.0, 1.0, 501)
    rows = []
    for tau in taus:
        frr = float((genuine  < tau).mean())   # genuine rejected
        far = float((impostor >= tau).mean())  # impostor accepted
        rows.append({"threshold": tau, "FRR": frr, "FAR": far})
    curve = pd.DataFrame(rows)

    # EER: threshold minimising |FAR - FRR|
    idx = int(np.argmin(np.abs(curve["FAR"] - curve["FRR"])))
    eer = float((curve.loc[idx, "FAR"] + curve.loc[idx, "FRR"]) / 2)
    return curve, eer, float(curve.loc[idx, "threshold"])


# --- EVALUATION ----------------------------------------------------------
def evaluate_model(X: np.ndarray, y: np.ndarray, label: str) -> dict:
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

    y_pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    classes = model.named_steps["clf"].classes_

    per_sat = argmax_metrics(y_te, y_pred)
    curve, eer, tau = threshold_sweep(y_te, proba, classes)

    return {
        "label": label,
        "accuracy": float((y_pred == y_te).mean()),
        "per_satellite": per_sat,
        "curve": curve,
        "eer": eer,
        "eer_threshold": tau,
        "mean_FRR": float(per_sat["FRR"].mean()),
        "mean_FAR": float(per_sat["FAR"].mean()),
        "n_features": X.shape[1],
    }

# --- PLOT ----------------------------------------------------------------
def plot_tradeoff(results: list[dict], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for res in results:
        c = res["curve"]
        ax1.plot(c["threshold"], c["FRR"], lw=1.6,
            label=f"{res['label']} - FRR")
        ax1.plot(c["threshold"], c["FAR"], lw=1.6, ls="--",
                label=f"{res['label']} - FAR")
    ax1.set_xlabel(r"Acceptance threshold $\tau$")
    ax1.set_ylabel("Error rate")
    ax1.set_title("Verification error rates against threshold",
            fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for res in results:
        c = res["curve"]
        ax2.plot(c["FAR"], c["FRR"], lw=1.8,
                label=f"{res['label']}  (EER {res['eer']:.1%})")
    ax2.plot([0, 1], [0, 1], color="grey", ls=":", lw=1,
            label="EER line")
    ax2.scatter([SATIQ_EER], [SATIQ_EER], s=90, color="crimson", zorder=5,
                label=f"SatIQ deep model (EER {SATIQ_EER:.1%})")
    ax2.set_xlabel("False Accept Rate")
    ax2.set_ylabel("False Reject Rate")
    ax2.set_title("Detection error trade-off", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Authentication error rates")
    print("=" * 72)

    df, features, rx = load_all()
    y = df["satellite_id"].to_numpy()

    results = [evaluate_model(clean(df[features].to_numpy()), y,
                            "28 waveform features")]
    if rx:
        results.append(evaluate_model(clean(df[rx].to_numpy()), y,
                                    "receiver-side metadata"))

    print("\n" + "-" * 72)
    print("A. ARGMAX DECISION RULE")
    print("-" * 72)
    for res in results:
        print(f"\n  {res['label']}  (accuracy {res['accuracy']:.1%})")
        print(f"    {'satellite':>10}{'genuine':>9}{'FRR':>9}{'FAR':>9}")
        for _, r in res["per_satellite"].iterrows():
            print(f"    {int(r['satellite']):>10}{int(r['n_genuine']):>9,}"
                f"{r['FRR']:>9.1%}{r['FAR']:>9.1%}")
        print(f"    {'mean':>10}{'':>9}{res['mean_FRR']:>9.1%}"
            f"{res['mean_FAR']:>9.1%}")

    print("\n" + "-" * 72)
    print("B. EQUAL ERROR RATE")
    print("-" * 72)
    print(f"\n  {'model':<26}{'EER':>9}{'at tau':>9}")
    for res in results:
        print(f"  {res['label']:<26}{res['eer']:>9.1%}"
            f"{res['eer_threshold']:>9.2f}")
    print(f"  {'SatIQ (published)':<26}{SATIQ_EER:>9.1%}{'-':>9}")

    best = min(results, key=lambda r: r["eer"])
    print(f"\n  Best classical model: {best['label']} at EER {best['eer']:.1%}")
    print(f"  SatIQ deep model:     EER {SATIQ_EER:.1%}")
    print(f"  Ratio: {best['eer'] / SATIQ_EER:.1f}x worse")

    print("\n" + "-" * 72)
    for res in results:
        fname = res["label"].replace(" ", "_")
        res["curve"].to_csv(OUT_TABLES / f"det_curve_{fname}.csv", index=False)

    print("Writing outputs...")
    out = []
    for res in results:
        for _, r in res["per_satellite"].iterrows():
            out.append({"model": res["label"], "satellite": int(r["satellite"]),
                        "n_genuine": int(r["n_genuine"]),
                        "FRR": r["FRR"], "FAR": r["FAR"],
                        "model_accuracy": res["accuracy"],
                        "model_EER": res["eer"]})
    pd.DataFrame(out).to_csv(OUT_TABLES / "authentication_metrics.csv",
                            index=False)
    plot_tradeoff(results, OUT_FIGURES / "authentication_tradeoff.png")

    md = """# Authentication error rates

## Why accuracy is the wrong metric

The supervisor's brief frames the security question as verification:

> If a signal claims to be satellite 1 but the RF fingerprint classifier
> does not classify it as satellite 1, then there may be spoofing.

That decision has two failure modes, and a control must be judged against
both. The False Reject Rate is the proportion of genuine traffic from a
satellite that fails to be recognised as that satellite, and would
therefore be flagged as spoofed. The False Accept Rate is the proportion of
messages from other transmitters that would be accepted under a claimed
identity.

Accuracy conceals this trade-off. A model at 27% accuracy rejects roughly
three quarters of authentic traffic, which no operator would deploy
irrespective of its resistance to attack.

## A. Argmax decision rule

| Model | Accuracy | Mean FRR | Mean FAR |
|-------|---------:|---------:|---------:|
"""
    for res in results:
        md += (f"| {res['label']} | {res['accuracy']:.1%} "
            f"| {res['mean_FRR']:.1%} | {res['mean_FAR']:.1%} |\n")

    md += """
Per-satellite detail:

| Model | Satellite | Genuine messages | FRR | FAR |
|-------|----------:|-----------------:|----:|----:|
"""
    for res in results:
        for _, r in res["per_satellite"].iterrows():
            md += (f"| {res['label']} | {int(r['satellite'])} "
                f"| {int(r['n_genuine']):,} | {r['FRR']:.1%} "
                f"| {r['FAR']:.1%} |\n")

    md += f"""
## B. Equal Error Rate

Random Forests emit class probabilities, so the verifier can accept a
claimed identity only when its probability exceeds a threshold tau.
Raising tau tightens acceptance: FAR falls while FRR rises. The Equal Error
Rate is the crossing point, and is the standard summary used in the
biometric and RF-fingerprinting literature.

| Model | EER | Threshold |
|-------|----:|----------:|
"""
    for res in results:
        md += (f"| {res['label']} | {res['eer']:.1%} "
            f"| {res['eer_threshold']:.2f} |\n")
    md += f"| SatIQ (Smailes et al., 2025) | {SATIQ_EER:.1%} | - |\n"

    md += f"""
The best classical model reaches an EER of {best['eer']:.1%} against
SatIQ's published {SATIQ_EER:.1%} on the same constellation --
approximately {best['eer'] / SATIQ_EER:.0f} times worse.

## C. Control assessment

At these error rates the control is unusable in both directions. Operating
at the equal error point would reject a large share of genuine satellite
traffic while admitting a comparable share of impersonations. Moving the
threshold to make either rate acceptable makes the other worse, and no
setting yields a workable operating point.

The gap to SatIQ is the substantive finding. Both approaches consume the
same physical-layer data from the same constellation. The difference lies
entirely in representation: hand-crafted summary statistics discard the
structure that a learned embedding preserves. Deep learning is not applied
here for its own sake but because the discriminative information is not
recoverable by simple aggregate statistics over the waveform.
"""
    (OUT_REPORTS / "authentication_metrics.md").write_text(md)

    print("  outputs/tables/authentication_metrics.csv")
    print("  outputs/figures/authentication_tradeoff.png")
    print("  outputs/reports/authentication_metrics.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
