"""
04_train_classifiers.py
=======================

PURPOSE
-------
Steps 8 and 9 of the 29 June task list.

Trains five simple classifiers on the 28 hand-crafted RF features and
compares them:
    1. Dummy (majority class)  -- the "chance level" reference
    2. Logistic Regression
    3. Decision Tree
    4. Random Forest
    5. Support Vector Machine (RBF)
    6. k-Nearest Neighbours
    7. Naive Bayes
    8. Simple feedforward neural network (MLPClassifier)

For each model it reports:
    - Accuracy (with a 95% bootstrap confidence interval)
    - Macro F1-score
    - Confusion matrix (saved as a figure)

WHY A DUMMY CLASSIFIER
----------------------
With 5 balanced classes, random guessing gives ~20% accuracy. The Dummy
classifier makes that baseline explicit, so any result can be honestly
compared against chance rather than against zero. This matters: a model
scoring 24% is NOT "learning a little" -- it is indistinguishable from
guessing, and the confidence intervals show that.

WHY A PIPELINE WITH StandardScaler
----------------------------------
Features have wildly different scales (mean_I ~ 0.001, bandwidth ~ 1e6).
Distance-based models (SVM, kNN) and gradient-based models (LogReg, MLP)
are dominated by large-scale features unless standardised.

Crucially, the scaler is fitted INSIDE a Pipeline, so it learns its mean
and standard deviation from the TRAINING fold only. Fitting the scaler on
the whole dataset before splitting would leak test-set information into
training -- a classic and often-undetected methodological error.

USAGE
-----
From the project root:
    python scripts/04_train_classifiers.py

Runtime: about 1-3 minutes.

OUTPUTS
-------
    outputs/tables/model_results.csv          -- the results table
    outputs/figures/confusion_matrices.png    -- all confusion matrices
    outputs/reports/model_comparison.md       -- dissertation-ready summary
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write figures to file without a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables"  / "features.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---------------------------------------------------------------
TEST_FRACTION  = 0.20      # 80% train / 20% test
RANDOM_SEED    = 42        # fixed so results are reproducible
N_BOOTSTRAP    = 1000      # resamples for the accuracy confidence interval


# --- STEP 1: LOAD THE FEATURE TABLE --------------------------------------
def load_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Read features.csv and split it into:
        X  -- the feature matrix, shape (n_messages, 28)
        y  -- the labels (satellite IDs), shape (n_messages,)

    'sample_id' is dropped: it is an index into the original dataset, not a
    physical property of the signal. Leaving it in would let a tree-based
    model memorise which row belongs to which satellite -- a leak, not a
    fingerprint.
    """
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURES_CSV}\n"
            f"Run scripts/03_extract_features.py first."
        )

    df = pd.read_csv(FEATURES_CSV)

    # Columns that are NOT physical properties of the signal. These must
    # never be fed to a model: global_index is the message's position in
    # the capture sequence, and because a satellite stays overhead for
    # minutes at a time, consecutive rows share a label. A tree splitting
    # on it learns arrival time, not transmitter identity.
    NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                           "Unnamed: 0", "satellite_id"}

    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    if len(feature_names) != 28:
        print(f"  WARNING: expected 28 features, found {len(feature_names)}")
        print(f"           {feature_names}")
    X = df[feature_names].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()

    return df, X, y, feature_names


# --- STEP 2: DEFINE THE MODELS -------------------------------------------
def build_models() -> dict[str, Pipeline]:
    """
    Each model is wrapped in a Pipeline so that StandardScaler is fitted on
    training data only (see the docstring at the top for why this matters).

    Tree-based models (Decision Tree, Random Forest) and Naive Bayes do not
    need scaling, but including it is harmless and keeps the code uniform.

    All hyperparameters are sklearn defaults except where a default would
    fail to converge on this data. Deliberately no tuning: the task list
    says the aim is something understandable, not something optimal.
    """
    return {
        # Predicts the most frequent class every time. This IS chance level.
        "Dummy (chance)": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   DummyClassifier(strategy="most_frequent")),
        ]),

        # Linear decision boundary in the 28-dimensional feature space.
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(max_iter=2000,
                                         random_state=RANDOM_SEED)),
        ]),

        # A single tree: interpretable, but prone to overfitting.
        "Decision Tree": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   DecisionTreeClassifier(random_state=RANDOM_SEED)),
        ]),

        # 100 trees voting. Also gives us feature importances (script 05).
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   RandomForestClassifier(n_estimators=100,
                                             random_state=RANDOM_SEED,
                                             n_jobs=-1)),
        ]),

        # Non-linear boundary via the RBF kernel. Needs scaled features.
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   SVC(kernel="rbf", random_state=RANDOM_SEED)),
        ]),

        # Classify by the 5 nearest training points in feature space.
        "k-NN (k=5)": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),

        # Assumes features are independent Gaussians per class.
        "Naive Bayes": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   GaussianNB()),
        ]),

        # The "simple feedforward neural network" the task list asks for:
        # one hidden layer of 64 units.
        "Neural Net (MLP)": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   MLPClassifier(hidden_layer_sizes=(64,),
                                    max_iter=500,
                                    random_state=RANDOM_SEED)),
        ]),
    }


# --- STEP 3: BOOTSTRAP CONFIDENCE INTERVAL -------------------------------
def bootstrap_accuracy_ci(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          n_resamples: int = N_BOOTSTRAP,
                          seed: int = RANDOM_SEED) -> tuple[float, float]:
    """
    Estimate a 95% confidence interval for accuracy by resampling the test
    set with replacement.

    Why this matters: if Random Forest scores 24.5% and k-NN scores 23.1%,
    it is tempting to say Random Forest "won". But with ~1,200 test samples
    the CI on each is roughly +/-2.5%, so those intervals overlap heavily
    and the difference is noise. Reporting intervals stops you from making
    a claim the data does not support -- examiners notice this.

    Returns:
        (lower_bound, upper_bound) at the 2.5th and 97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)

    for i in range(n_resamples):
        # Draw n indices WITH replacement -- one simulated "alternative"
        # test set drawn from the same distribution.
        idx = rng.integers(0, n, size=n)
        scores[i] = accuracy_score(y_true[idx], y_pred[idx])

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# --- STEP 4: TRAIN AND EVALUATE ------------------------------------------
def evaluate_model(name: str,
                   pipeline: Pipeline,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test:  np.ndarray, y_test:  np.ndarray) -> dict:
    """Fit one model and compute all the metrics for it."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    ci_low, ci_high = bootstrap_accuracy_ci(y_test, y_pred)

    return {
        "model":            name,
        "accuracy":         accuracy,
        "accuracy_ci_low":  ci_low,
        "accuracy_ci_high": ci_high,
        "macro_f1":         macro_f1,
        "confusion":        confusion_matrix(y_test, y_pred),
        "y_pred":           y_pred,
    }


# --- STEP 5: PLOT CONFUSION MATRICES -------------------------------------
def plot_confusion_matrices(results: list[dict],
                            class_labels: np.ndarray,
                            output_path: Path) -> None:
    """
    Draw every model's confusion matrix in one grid figure.

    Rows are the TRUE satellite, columns are the PREDICTED satellite.
    A model that works shows a bright diagonal. A model at chance level
    shows either a uniform grid or one bright column (everything predicted
    as a single class).
    """
    n_models = len(results)
    n_cols   = 4
    n_rows   = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, res in zip(axes, results):
        cm = res["confusion"]
        # Normalise each row so colours mean "% of this satellite's messages"
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm / np.maximum(row_sums, 1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{res['model']}\nacc={res['accuracy']:.3f}", fontsize=10)
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        # Write the raw counts into each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")

    # Hide any unused subplot slots
    for ax in axes[n_models:]:
        ax.axis("off")

    fig.suptitle("Confusion matrices - 5-satellite classification",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- STEP 6: WRITE THE REPORT --------------------------------------------
def write_markdown_report(results: list[dict],
                          class_labels: np.ndarray,
                          n_train: int,
                          n_test: int,
                          n_features: int,
                          chance_level: float,
                          output_path: Path) -> None:
    """Write a dissertation-ready summary of the comparison."""
    md = f"""# Baseline classifier comparison

## Experimental setup

- **Task:** multi-class classification of Iridium satellite transmitter ID
- **Classes:** {len(class_labels)} satellites ({', '.join(str(c) for c in class_labels)})
- **Features:** {n_features} hand-crafted RF features per message
- **Split:** stratified {int((1-TEST_FRACTION)*100)}% train / {int(TEST_FRACTION*100)}% test
- **Training samples:** {n_train:,}
- **Test samples:** {n_test:,}
- **Chance level (majority class):** {chance_level:.1%}
- **Preprocessing:** StandardScaler fitted inside a Pipeline on training folds only
- **Hyperparameters:** scikit-learn defaults (no tuning at this stage)

## Results

| Model | Accuracy | 95% CI | Macro F1 | Notes |
|-------|---------:|:------:|---------:|-------|
"""
    for res in results:
        note = "chance reference" if "Dummy" in res["model"] else ""
        md += (f"| {res['model']} "
               f"| {res['accuracy']:.4f} "
               f"| [{res['accuracy_ci_low']:.3f}, {res['accuracy_ci_high']:.3f}] "
               f"| {res['macro_f1']:.4f} "
               f"| {note} |\n")

    # Work out whether anything actually beat chance
    dummy = next((r for r in results if "Dummy" in r["model"]), None)
    best  = max((r for r in results if "Dummy" not in r["model"]),
                key=lambda r: r["accuracy"])

    beats_chance = (dummy is not None
                    and best["accuracy_ci_low"] > dummy["accuracy_ci_high"])

    md += f"""

## Interpretation

The strongest model is **{best['model']}** at {best['accuracy']:.1%} accuracy
(95% CI [{best['accuracy_ci_low']:.1%}, {best['accuracy_ci_high']:.1%}]),
compared with a chance level of {chance_level:.1%}.

"""
    if beats_chance:
        md += """The best model's confidence interval sits above the chance
reference, so the hand-crafted features do carry some class-discriminative
information. The margin should still be interpreted alongside the feature
importance analysis (Section X) to determine whether the signal comes from
hardware characteristics or from channel and geometry effects.
"""
    else:
        md += """**No model's confidence interval separates from the chance
reference.** The hand-crafted feature set, as extracted, does not carry
usable class-discriminative information for this task.

This is a genuine empirical finding rather than an implementation error, and
it is diagnosed further in Section X (channel-dominance analysis), which
quantifies how much of each feature's variance is explained by received
signal strength rather than by transmitter identity. Overlapping confidence
intervals also mean that ranking the models against one another here would
not be statistically supportable.
"""

    md += """
## Methodological notes

- Confidence intervals are computed by bootstrap resampling of the test set
  (1,000 resamples, percentile method). Where intervals overlap, differences
  between models are not statistically distinguishable.
- The train/test split is random over messages. Messages captured during the
  same satellite pass share channel conditions, so a random split may allow a
  model to exploit pass-level rather than transmitter-level information. A
  pass-aware (GroupKFold) evaluation is reported separately.
"""

    with open(output_path, "w") as f:
        f.write(md)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Baseline classifier comparison")
    print("=" * 70)

    # 1. Load
    print("\nStep 1 - Loading features...")
    df, X, y, feature_names = load_features()
    class_labels = np.unique(y)
    print(f"  Messages:  {len(y):,}")
    print(f"  Features:  {len(feature_names)}")
    print(f"  Classes:   {len(class_labels)}  ({', '.join(str(c) for c in class_labels)})")
    print("  Class distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        print(f"    Sat {int(label):>3d}: {count:,} messages")

    # Chance level = proportion of the most common class
    chance_level = float(np.max(np.bincount(
        pd.factorize(y)[0])) / len(y))
    print(f"  Chance level (majority class): {chance_level:.1%}")

    # 2. Split
    print("\nStep 2 - Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,           # keeps class proportions identical in both sets
    )
    print(f"  Train: {len(y_train):,}   Test: {len(y_test):,}")

    # 3. Train and evaluate
    print("\nStep 3 - Training models...")
    models  = build_models()
    results = []
    for name, pipeline in models.items():
        print(f"  {name:<22s} ... ", end="", flush=True)
        res = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(res)
        print(f"acc={res['accuracy']:.4f}  "
              f"CI=[{res['accuracy_ci_low']:.3f},{res['accuracy_ci_high']:.3f}]  "
              f"F1={res['macro_f1']:.4f}")

    # 4. Results table
    print("\nStep 4 - Writing results table...")
    table = pd.DataFrame([{
        "model":            r["model"],
        "accuracy":         round(r["accuracy"], 4),
        "accuracy_ci_low":  round(r["accuracy_ci_low"], 4),
        "accuracy_ci_high": round(r["accuracy_ci_high"], 4),
        "macro_f1":         round(r["macro_f1"], 4),
        "n_train":          len(y_train),
        "n_test":           len(y_test),
        "n_features":       len(feature_names),
        "n_classes":        len(class_labels),
    } for r in results])
    table_path = OUT_TABLES / "model_results.csv"
    table.to_csv(table_path, index=False)
    print(f"  {table_path.relative_to(PROJECT_ROOT)}")

    # 5. Confusion matrices
    print("\nStep 5 - Plotting confusion matrices...")
    fig_path = OUT_FIGURES / "confusion_matrices.png"
    plot_confusion_matrices(results, class_labels, fig_path)
    print(f"  {fig_path.relative_to(PROJECT_ROOT)}")

    # 6. Report
    print("\nStep 6 - Writing markdown report...")
    report_path = OUT_REPORTS / "model_comparison.md"
    write_markdown_report(results, class_labels,
                          len(y_train), len(y_test),
                          len(feature_names), chance_level,
                          report_path)
    print(f"  {report_path.relative_to(PROJECT_ROOT)}")

    # 7. Per-class detail for the best non-dummy model
    best = max((r for r in results if "Dummy" not in r["model"]),
               key=lambda r: r["accuracy"])
    print("\n" + "=" * 70)
    print(f"Best non-trivial model: {best['model']}")
    print("=" * 70)
    print(classification_report(y_test, best["y_pred"], zero_division=0))

    print("=" * 70)
    print("Done. Next: scripts/05_feature_importance.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
