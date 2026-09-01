"""Final Chapter 5 validation for the v1 RF baseline.

Run from the project root:
    python scripts/33_ch5_final_validation.py

Assumptions frozen from the final fusion evidence adapter:
- outputs/tables/features.csv
- 28 v1 waveform features
- stratified 80/20 split
- random_state=42
- RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
- no scaling

Outputs are written to outputs/ch5_final_validation/ so older results are not overwritten.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "outputs" / "tables" / "features.csv"
OUT = ROOT / "outputs" / "ch5_final_validation"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
for d in (TABLES, FIGURES, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20
N_TREES = 200
N_BOOT = 1000
NON_FEATURE = {"sample_id", "global_index", "index", "Unnamed: 0", "satellite_id"}


def bootstrap_ci(correct, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(correct)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        ix = rng.integers(0, m, size=m)
        scores[i] = correct[ix].mean()
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def exact_mcnemar(correct_a, correct_b):
    # A wrong/B right and A right/B wrong
    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)
    return n01, n10, p


def rf(**kwargs):
    params = dict(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    params.update(kwargs)
    return RandomForestClassifier(**params)


def main():
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing {FEATURES}")

    df = pd.read_csv(FEATURES)
    feat = [c for c in df.columns if c not in NON_FEATURE]
    if len(feat) != 28:
        raise ValueError(f"Expected 28 v1 features, found {len(feat)}")

    X = df[feat].to_numpy(float)
    y = df["satellite_id"].to_numpy()
    idx = np.arange(len(y))

    itr, ite = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]

    # Re-train the frozen RF configuration for consistency checking.
    model = rf()
    model.fit(Xtr, ytr)
    pred_retrained = model.predict(Xte)
    retrained_accuracy = accuracy_score(yte, pred_retrained)

    # For statistical validation, prefer the saved final fusion evidence
    # predictions if present. This prevents a scikit-learn version change from
    # silently altering the already-frozen Chapter 5 baseline.
    evidence_candidates = [
        ROOT / "fusion" / "outputs" / "tables" / "evidence.csv",
        ROOT / "evidence.csv",
    ]
    evidence_path = next((q for q in evidence_candidates if q.exists()), None)
    if evidence_path is not None:
        ev = pd.read_csv(evidence_path)
        required = {"source_row_index", "true_satellite", "rf_predicted_sat"}
        if not required.issubset(ev.columns):
            raise ValueError(f"Evidence file lacks required columns: {evidence_path}")
        if len(ev) != len(ite) or not np.array_equal(ev["source_row_index"].to_numpy(), ite):
            raise ValueError("Saved evidence does not match the frozen stratified 80/20 test split")
        if not np.array_equal(ev["true_satellite"].to_numpy(), yte):
            raise ValueError("Saved evidence true labels do not match features.csv")
        pred = ev["rf_predicted_sat"].to_numpy()
        evidence_accuracy = accuracy_score(yte, pred)
        evidence_source = str(evidence_path)
    else:
        pred = pred_retrained
        evidence_accuracy = retrained_accuracy
        evidence_source = "retrained in this run (saved evidence.csv not found)"

    correct_rf = pred == yte

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    pred_dummy = dummy.predict(Xte)
    correct_dummy = pred_dummy == yte

    rf_ci = bootstrap_ci(correct_rf)
    d_ci = bootstrap_ci(correct_dummy)
    n01, n10, p = exact_mcnemar(correct_dummy, correct_rf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf(), Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=1)

    baseline = pd.DataFrame([
        {
            "model": "Random Forest",
            "n_test": len(yte),
            "correct": int(correct_rf.sum()),
            "accuracy": correct_rf.mean(),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "bootstrap_ci_low": rf_ci[0],
            "bootstrap_ci_high": rf_ci[1],
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
        },
        {
            "model": "Most-frequent baseline",
            "n_test": len(yte),
            "correct": int(correct_dummy.sum()),
            "accuracy": correct_dummy.mean(),
            "macro_f1": f1_score(yte, pred_dummy, average="macro", zero_division=0),
            "bootstrap_ci_low": d_ci[0],
            "bootstrap_ci_high": d_ci[1],
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
        },
    ])
    baseline.to_csv(TABLES / "baseline_statistical_validation.csv", index=False)

    pd.DataFrame([{
        "comparison": "Random Forest vs most-frequent baseline",
        "dummy_wrong_rf_right": n01,
        "dummy_right_rf_wrong": n10,
        "discordant_total": n01 + n10,
        "exact_mcnemar_p": p,
        "significant_at_0.05": p < 0.05,
    }]).to_csv(TABLES / "mcnemar_final.csv", index=False)

    # Learning curve: fixed held-out test set, stratified training subsets.
    curve = []
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        if frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=SEED)
            sub, _ = next(sss.split(Xtr, ytr))
        else:
            sub = np.arange(len(ytr))
        m = rf().fit(Xtr[sub], ytr[sub])
        acc = accuracy_score(yte, m.predict(Xte))
        d = DummyClassifier(strategy="most_frequent").fit(Xtr[sub], ytr[sub])
        chance = accuracy_score(yte, d.predict(Xte))
        curve.append({"train_fraction": frac, "n_train": len(sub), "accuracy": acc, "chance": chance})
    curve = pd.DataFrame(curve)
    curve.to_csv(TABLES / "learning_curve_final.csv", index=False)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(curve["n_train"], curve["accuracy"], marker="o", label="Random Forest")
    plt.plot(curve["n_train"], curve["chance"], linestyle="--", label="Most-frequent baseline")
    plt.xlabel("Training messages")
    plt.ylabel("Held-out accuracy")
    plt.title("Learning Curve – Final v1 Random Forest Configuration")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "learning_curve_final.png", dpi=200)
    plt.close()

    # Hyperparameter sensitivity: select using training-only 3-fold CV,
    # evaluate only the best candidate once on the held-out test set.
    configs = [
        ("final_default", {}),
        ("100_trees", {"n_estimators": 100}),
        ("300_trees", {"n_estimators": 300}),
        ("max_depth_8", {"max_depth": 8}),
        ("min_samples_leaf_5", {"min_samples_leaf": 5}),
    ]
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    hp_rows = []
    for name, params in configs:
        m = rf(**params)
        scores = cross_val_score(m, Xtr, ytr, cv=cv3, scoring="accuracy", n_jobs=1)
        hp_rows.append({
            "configuration": name,
            "params": json.dumps(params, sort_keys=True),
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
        })
    hp = pd.DataFrame(hp_rows).sort_values("cv_accuracy_mean", ascending=False)
    best_name = hp.iloc[0]["configuration"]
    best_params = dict(configs[[x[0] for x in configs].index(best_name)][1])
    best_model = rf(**best_params).fit(Xtr, ytr)
    best_test = accuracy_score(yte, best_model.predict(Xte))
    hp["heldout_test_accuracy"] = np.nan
    hp.loc[hp["configuration"] == best_name, "heldout_test_accuracy"] = best_test
    hp.to_csv(TABLES / "hyperparameter_sensitivity_final.csv", index=False)

    report = f"# Chapter 5 final RF validation\n\n"
    report += f"Statistical-prediction source: {evidence_source}\n\n"
    report += f"Saved/frozen RF accuracy used for statistics: {evidence_accuracy:.4f}\n\n"
    report += f"Current-environment retrained RF accuracy: {retrained_accuracy:.4f}\n\n"
    report += f"Difference: {retrained_accuracy - evidence_accuracy:+.6f}\n\n"
    report += f"RF test accuracy: {correct_rf.mean():.4f} ({int(correct_rf.sum())}/{len(yte)})\n\n"
    report += f"RF macro-F1: {f1_score(yte, pred, average='macro'):.4f}\n\n"
    report += f"RF 95% bootstrap CI: [{rf_ci[0]:.4f}, {rf_ci[1]:.4f}]\n\n"
    report += f"Most-frequent accuracy: {correct_dummy.mean():.4f}\n\n"
    report += f"Most-frequent 95% bootstrap CI: [{d_ci[0]:.4f}, {d_ci[1]:.4f}]\n\n"
    report += f"Exact McNemar p-value: {p:.6f} (discordant: {n01} vs {n10})\n\n"
    report += f"5-fold training-set CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
    report += f"Best sensitivity configuration by training-only CV: {best_name} {best_params}\n\n"
    report += f"Held-out test accuracy of selected sensitivity configuration: {best_test:.4f}\n"
    (REPORTS / "ch5_final_validation.md").write_text(report)

    print(report)
    print(f"Outputs written to: {OUT}")


if __name__ == "__main__":
    main()
