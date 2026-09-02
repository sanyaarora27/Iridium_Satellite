"""
29_feature_group_ablation.py
============================

Controlled feature-group ablation for the primary 28-feature RF model.

Purpose
-------
Answer the supervisor's Action 6:
1. Train with all 28 features.
2. Remove predefined feature groups one at a time.
3. Report accuracy, macro-F1, probability of detection,
   false rejection / "false alarm" probability, and false acceptance rate.
4. Evaluate a top-10 feature subset.
5. Keep the same split, seed, model, and preprocessing for every experiment.

Important methodological controls
---------------------------------
- Same stratified 80/20 split for every experiment.
- random_state = 42.
- Random Forest = 200 trees, matching the primary classical baseline.
- Median imputation is fitted on TRAINING DATA ONLY.
- global_index / satellite_id are never model inputs.
- timestamp_global is not present in features.csv and is therefore not used.
- Top-10 feature selection is performed on TRAINING DATA ONLY.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# PATHS / CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_TREES = 200

NON_FEATURE_COLUMNS = {
    "sample_id",
    "global_index",
    "index",
    "Unnamed: 0",
    "satellite_id",
}

# PREDEFINED FEATURE GROUPS
#
# Groups may overlap intentionally.
# Example: std_I is both a time-domain statistic and strongly amplitude
# dependent. Each ablation asks a different question.

TIME_DOMAIN = [
    "mean_I", "mean_Q",
    "var_I", "var_Q",
    "std_I", "std_Q",
    "max_I", "max_Q",
    "min_I", "min_Q",
    "skew_I", "skew_Q",
    "kurt_I", "kurt_Q",
    "median_I", "median_Q",
    "iqr_I", "iqr_Q",
]

AMPLITUDE_POWER = [
    "var_I", "var_Q",
    "std_I", "std_Q",
    "max_I", "max_Q",
    "min_I", "min_Q",
    "iqr_I", "iqr_Q",
    "signal_power",
    "papr",
]

FREQUENCY_DOMAIN = [
    "fft_mean_magnitude",
    "peak_frequency",
    "spectral_centroid",
    "bandwidth",
    "occupied_bandwidth",
]

IQ_TEMPORAL = [
    "iq_ratio",
    "iq_correlation",
    "zero_crossing_rate",
]

# HELPERS

def make_model() -> Pipeline:
    """
    Median imputation is train-fitted because it lives inside the pipeline.
    RF itself does not require scaling.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

def prepare_X(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """
    Convert selected columns to numeric matrix.
    Convert +/- infinity to NaN so the train-fitted imputer can handle them.
    """
    X = df[features].astype(float).replace(
        [np.inf, -np.inf], np.nan
    )
    return X.to_numpy()

def authentication_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """
    Claimed-identity interpretation under the argmax decision rule.

    For each satellite T:
      FRR(T) = genuine T observations not predicted as T
      FAR(T) = non-T observations predicted as T

    Mean values are macro-averaged across the five satellites.

    Detection probability:
      P_D = 1 - mean(FRR)

    Supervisor's wording uses 'false alarm probability' for genuine
    signals wrongly rejected. That corresponds here to FRR.

    We additionally report FAR explicitly.
    """
    frrs = []
    fars = []

    for sat in np.unique(y_true):
        genuine = y_true == sat
        impostor = ~genuine

        frr = float((y_pred[genuine] != sat).mean())
        far = float((y_pred[impostor] == sat).mean())

        frrs.append(frr)
        fars.append(far)

    mean_frr = float(np.mean(frrs))
    mean_far = float(np.mean(fars))
    pdetection = 1.0 - mean_frr

    return pdetection, mean_frr, mean_far

def eer_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float]:
    """
    Optional EER metric for each ablation.

    Genuine score:
        probability assigned to the true/claimed identity.

    Impostor scores:
        probabilities assigned to every incorrect claimed identity.
    """
    genuine_scores = []
    impostor_scores = []

    for i, true_label in enumerate(y_true):
        for j, cls in enumerate(classes):
            if cls == true_label:
                genuine_scores.append(probabilities[i, j])
            else:
                impostor_scores.append(probabilities[i, j])

    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    thresholds = np.linspace(0.0, 1.0, 501)

    best_difference = np.inf
    best_eer = np.nan
    best_threshold = np.nan

    for tau in thresholds:
        frr = float((genuine_scores < tau).mean())
        far = float((impostor_scores >= tau).mean())

        difference = abs(frr - far)

        if difference < best_difference:
            best_difference = difference
            best_eer = (frr + far) / 2.0
            best_threshold = tau

    return float(best_eer), float(best_threshold)

def evaluate(
    df: pd.DataFrame,
    features: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
) -> dict:

    X = prepare_X(df, features)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["clf"].classes_

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    pdetection, frr, far = authentication_metrics(y_test, y_pred)
    eer, eer_threshold = eer_metrics(y_test, probabilities, classes)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "detection_probability": float(pdetection),
        "false_alarm_probability": float(frr),
        "false_rejection_rate": float(frr),
        "false_acceptance_rate": float(far),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
    }

def select_top10_training_only(
    df: pd.DataFrame,
    all_features: list[str],
    train_idx: np.ndarray,
    y: np.ndarray,
) -> tuple[list[str], pd.DataFrame]:
    """
    Rank features using only the training partition.
    This prevents test-set feature-selection leakage.
    """
    X = prepare_X(df, all_features)

    X_train = X[train_idx]
    y_train = y[train_idx]

    model = make_model()
    model.fit(X_train, y_train)

    importances = model.named_steps["clf"].feature_importances_

    table = (
        pd.DataFrame({
            "feature": all_features,
            "training_only_importance": importances,
        })
        .sort_values("training_only_importance", ascending=False)
        .reset_index(drop=True)
    )

    return table.head(10)["feature"].tolist(), table

# MAIN

def main() -> None:

    print("=" * 80)
    print("ACTION 6 - FEATURE GROUP ABLATION")
    print("=" * 80)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run 04_extract_features.py first."
        )

    df = pd.read_csv(FEATURES_CSV)

    all_features = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    y = df["satellite_id"].to_numpy()

    print(f"\nMessages: {len(df):,}")
    print(f"Classes:  {sorted(np.unique(y).tolist())}")
    print(f"Features: {len(all_features)}")

    if len(all_features) != 28:
        raise ValueError(
            f"Expected 28 v1 RF features, found {len(all_features)}"
        )

    # Validate predefined feature groups.
    for group_name, group in {
        "TIME_DOMAIN": TIME_DOMAIN,
        "AMPLITUDE_POWER": AMPLITUDE_POWER,
        "FREQUENCY_DOMAIN": FREQUENCY_DOMAIN,
        "IQ_TEMPORAL": IQ_TEMPORAL,
    }.items():
        missing = sorted(set(group) - set(all_features))
        if missing:
            raise KeyError(
                f"{group_name} contains missing features: {missing}"
            )

        # One split created ONCE and reused for every ablation.
    
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"Train: {len(train_idx):,} | Test: {len(test_idx):,} "
        f"| seed={RANDOM_SEED}"
    )

        # Top-10 selection from TRAINING DATA ONLY.
    
    top10, ranking = select_top10_training_only(
        df, all_features, train_idx, y
    )

    ranking_path = (
        OUT_TABLES / "ablation_training_only_feature_importance.csv"
    )
    ranking.to_csv(ranking_path, index=False)

    print("\nTraining-only top 10 features:")
    for rank, name in enumerate(top10, start=1):
        imp = ranking.loc[
            ranking["feature"] == name,
            "training_only_importance"
        ].iloc[0]
        print(f"  {rank:2d}. {name:<25s} {imp:.5f}")

        # Experiments
    
    experiments = [
        {
            "experiment": "All 28 v1 features",
            "features": all_features,
            "removed": [],
            "interpretation":
                "Primary waveform-feature baseline; leakage metadata excluded.",
        },
        {
            "experiment": "Without time-domain statistics",
            "features": [
                f for f in all_features if f not in TIME_DOMAIN
            ],
            "removed": TIME_DOMAIN,
            "interpretation":
                "Tests dependence on basic I/Q distribution statistics.",
        },
        {
            "experiment": "Without amplitude/power",
            "features": [
                f for f in all_features if f not in AMPLITUDE_POWER
            ],
            "removed": AMPLITUDE_POWER,
            "interpretation":
                "Tests dependence on received-amplitude/power-sensitive information.",
        },
        {
            "experiment": "Without frequency-domain",
            "features": [
                f for f in all_features if f not in FREQUENCY_DOMAIN
            ],
            "removed": FREQUENCY_DOMAIN,
            "interpretation":
                "Tests contribution of FFT, frequency-offset and bandwidth descriptors.",
        },
        {
            "experiment": "Without I/Q relationship + temporal",
            "features": [
                f for f in all_features if f not in IQ_TEMPORAL
            ],
            "removed": IQ_TEMPORAL,
            "interpretation":
                "Tests I/Q imbalance/correlation and zero-crossing information.",
        },
        {
            "experiment": "Top-10 RF features only",
            "features": top10,
            "removed": [
                f for f in all_features if f not in top10
            ],
            "interpretation":
                "Tests whether a compact training-selected subset retains performance.",
        },
    ]

    rows = []

    print("\n" + "-" * 80)
    print("Running ablations")
    print("-" * 80)

    for exp in experiments:

        metrics = evaluate(
            df,
            exp["features"],
            train_idx,
            test_idx,
            y,
        )

        row = {
            "experiment": exp["experiment"],
            "n_features": len(exp["features"]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "detection_probability": metrics["detection_probability"],
            "false_alarm_probability": metrics["false_alarm_probability"],
            "false_rejection_rate": metrics["false_rejection_rate"],
            "false_acceptance_rate": metrics["false_acceptance_rate"],
            "eer": metrics["eer"],
            "eer_threshold": metrics["eer_threshold"],
            "features_used": "; ".join(exp["features"]),
            "features_removed": "; ".join(exp["removed"]),
            "interpretation": exp["interpretation"],
        }

        rows.append(row)

        print(
            f"\n{exp['experiment']}"
            f"\n  n features : {len(exp['features'])}"
            f"\n  accuracy   : {metrics['accuracy']:.4f}"
            f"\n  macro F1   : {metrics['macro_f1']:.4f}"
            f"\n  P_D        : {metrics['detection_probability']:.4f}"
            f"\n  FRR/P_FA   : {metrics['false_rejection_rate']:.4f}"
            f"\n  FAR        : {metrics['false_acceptance_rate']:.4f}"
            f"\n  EER        : {metrics['eer']:.4f}"
        )

    results = pd.DataFrame(rows)

        # Deltas relative to all-feature baseline.
    
    baseline = results.iloc[0]

    results["delta_accuracy_pp"] = (
        results["accuracy"] - baseline["accuracy"]
    ) * 100.0

    results["delta_macro_f1_pp"] = (
        results["macro_f1"] - baseline["macro_f1"]
    ) * 100.0

    results["delta_detection_pp"] = (
        results["detection_probability"]
        - baseline["detection_probability"]
    ) * 100.0

    results["delta_FAR_pp"] = (
        results["false_acceptance_rate"]
        - baseline["false_acceptance_rate"]
    ) * 100.0

        # Save table
    
    table_path = OUT_TABLES / "feature_group_ablation.csv"
    results.to_csv(table_path, index=False)

    # Compact supervisor table
    compact_columns = [
        "experiment",
        "n_features",
        "accuracy",
        "macro_f1",
        "detection_probability",
        "false_alarm_probability",
        "false_acceptance_rate",
        "eer",
        "delta_accuracy_pp",
        "delta_macro_f1_pp",
        "interpretation",
    ]

    compact = results[compact_columns].copy()

    compact_path = OUT_TABLES / "feature_group_ablation_summary.csv"
    compact.to_csv(compact_path, index=False)

        # Plot: accuracy and macro-F1
    
    labels = results["experiment"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        results["accuracy"],
        width,
        label="Accuracy",
    )
    ax.bar(
        x + width / 2,
        results["macro_f1"],
        width,
        label="Macro F1",
    )

    ax.set_ylabel("Score")
    ax.set_title("Random Forest Feature-Group Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    figure_path = OUT_FIGURES / "feature_group_ablation.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

        # Markdown report
    
    report_path = OUT_REPORTS / "feature_group_ablation.md"

    lines = [
        "# Feature-group ablation",
        "",
        "## Method",
        "",
        (
            "All experiments used the same stratified 80/20 split "
            f"(random seed {RANDOM_SEED}) and a {N_TREES}-tree Random Forest."
        ),
        (
            "Median imputation was fitted only on the training partition. "
            "`global_index`, `satellite_id`, and chronology-derived metadata "
            "were not predictive inputs."
        ),
        (
            "The top-10 subset was selected using Random Forest importance "
            "computed on the training partition only."
        ),
        "",
        "No explicit raw-phase feature group exists in the primary v1 "
        "28-feature representation; phase/CFO-oriented descriptors belong "
        "to later feature experiments.",
        "",
        "## Results",
        "",
        "| Experiment | n | Accuracy | Macro F1 | P_D | FRR / P_FA | FAR | EER | Δ Acc. pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in results.iterrows():
        lines.append(
            f"| {r['experiment']} "
            f"| {int(r['n_features'])} "
            f"| {r['accuracy']:.3f} "
            f"| {r['macro_f1']:.3f} "
            f"| {r['detection_probability']:.3f} "
            f"| {r['false_rejection_rate']:.3f} "
            f"| {r['false_acceptance_rate']:.3f} "
            f"| {r['eer']:.3f} "
            f"| {r['delta_accuracy_pp']:+.2f} |"
        )

    lines.extend([
        "",
        "## Training-only top 10 features",
        "",
    ])

    for i, feature in enumerate(top10, start=1):
        importance = ranking.loc[
            ranking["feature"] == feature,
            "training_only_importance"
        ].iloc[0]

        lines.append(
            f"{i}. `{feature}` — importance {importance:.5f}"
        )

    lines.extend([
        "",
        "## Interpretation rule",
        "",
        (
            "A reduction after removing a feature group indicates that the "
            "group contributed useful predictive information. It does not, "
            "by itself, establish that the information is transmitter-hardware "
            "specific. The result must be interpreted alongside the existing "
            "channel-dominance and leakage analyses."
        ),
    ])

    report_path.write_text("\n".join(lines))

        # Console summary
    
    print("\n" + "=" * 80)
    print("FINAL ABLATION SUMMARY")
    print("=" * 80)

    display_cols = [
        "experiment",
        "n_features",
        "accuracy",
        "macro_f1",
        "detection_probability",
        "false_rejection_rate",
        "false_acceptance_rate",
        "eer",
        "delta_accuracy_pp",
    ]

    print(results[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {table_path}")
    print(f"  {compact_path}")
    print(f"  {ranking_path}")
    print(f"  {figure_path}")
    print(f"  {report_path}")

    print("\nNOTE:")
    print(
        "  Primary v1 already excludes leakage metadata. "
        "There is therefore no separate 'remove metadata' "
        "ablation from the 28-feature input."
    )
    print(
        "  No explicit phase statistics exist in v1, so no "
        "phase-ablation result is fabricated."
    )

if __name__ == "__main__":
    main()
