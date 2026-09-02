"""
05_train_classifiers.py
=======================

Steps 8-10 of the supervisor's 29 June task list:
  Step 8  - train the four required classical classifiers on features.csv
            (plus three deliberate additions: a most-frequent-class
            baseline as the scientific floor, a single Decision Tree to
            quantify what the Random Forest's ensembling buys, and
            Gaussian Naive Bayes to test whether feature interactions
            matter - see MODELS below for the full rationale)
  Step 9  - compare them fairly on held-out data
  Step 10 - rank which features matter most

INPUT
-----
    outputs/tables/features.csv   (produced by 04_extract_features.py)

OUTPUTS
-------
    outputs/tables/classifier_comparison.csv   accuracy + macro-F1 per model
    outputs/reports/classification_reports.txt per-class precision/recall
    outputs/figures/confusion_matrices.png     2x2 grid, one per classifier
    outputs/figures/feature_importance.png     top features (Random Forest)
    outputs/tables/feature_importance.csv      full ranking

METHOD, AND WHY
---------------
1. STRATIFIED 80/20 TRAIN/TEST SPLIT.
   Stratified = each satellite keeps the same proportion in train and
   test. The test set is touched exactly once, at the end - it simulates
   "messages the model has never seen". random_state is fixed so the
   split (and every result) is reproducible.

2. 5-FOLD CROSS-VALIDATION ON THE TRAINING SET.
   A single accuracy number can be lucky. CV trains each model 5 times
   on different 80% slices of the training data and reports the mean
   and standard deviation - so we can tell whether the difference
   between two classifiers is real or noise.

3. FEATURE SCALING - BUT ONLY FOR THE MODELS THAT NEED IT.
   Logistic Regression, SVM, and k-NN all compare or combine feature
   values directly (distances, dot products). Our features live on
   wildly different scales - signal_power ~1e-3 vs peak_frequency ~1e6
   - so without scaling, the largest-scale feature silently dominates.
   StandardScaler transforms each feature to mean 0, std 1.
   Random Forest splits one feature at a time at learned thresholds, so
   scaling changes nothing for it - we leave it unscaled to prove we
   know WHY we scale, not because it's a ritual.
   The scaler lives INSIDE a Pipeline, so during cross-validation it is
   re-fitted on each training fold only. Scaling before splitting would
   leak test-set statistics (mean/std) into training - a classic subtle
   error called data leakage.

USAGE
-----
From the project root:
    python scripts/05_train_classifiers.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
FEATURES_CSV   = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUTPUT_TABLES  = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
for d in (OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42     # fixed seed -> reproducible split and models
TEST_FRACTION = 0.20  # 80% train / 20% test
CV_FOLDS = 5
N_BOOTSTRAP = 1000

# The four classifiers required by the task list, plus three deliberate
# additions that each answer a specific question:
#
#   - Dummy (most frequent): the scientific FLOOR. Predicts the largest
#     class every time. Any model that can't beat this has learned
#     nothing. Puts an explicit chance baseline in the comparison table.
#   - Decision Tree (single): one tree vs the Random Forest's 200.
#     The gap between them IS the variance reduction from bagging -
#     it demonstrates why ensembles exist, with real numbers.
#   - Gaussian Naive Bayes: assumes features are INDEPENDENT given the
#     class - ours provably aren't (std_I = sqrt(var_I)). If it scores
#     well, per-feature statistics alone carry the fingerprint; if it
#     scores poorly, feature interactions matter. Either way, a finding.
#
# Each entry: (display name, estimator, needs_scaling)
#
# Hyperparameters are deliberately the SIMPLE, defensible defaults:
#   - LogisticRegression: max_iter raised so the solver converges on
#     28 standardised features; everything else default.
#   - RandomForest: 200 trees (default 100 is fine; 200 slightly
#     stabilises feature importances at negligible cost).
#   - SVC: RBF kernel (default) - the standard nonlinear baseline.
#   - kNN: k=5 (default) - odd k avoids ties.
#   - DecisionTree: default (fully grown) - the point is to show a
#     single unregularised tree overfitting relative to the forest.
#   - GaussianNB / Dummy: no hyperparameters worth touching.
# No hyperparameter tuning at this stage: the goal is an honest
# baseline comparison, not a leaderboard. Tuning can be a later step
# if the supervisor asks for it.
#
# Scaling notes for the additions: tree splits (Decision Tree) and
# per-feature Gaussians (NB) are scale-invariant, and Dummy ignores
# features entirely - so none of the three needs the scaler.
MODELS = [
    ("Baseline (most frequent)",
     DummyClassifier(strategy="most_frequent"), False),
    ("Logistic Regression",
     LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True),
    ("Decision Tree",
     DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    ("Random Forest",
     RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                            n_jobs=-1), False),
    ("SVM (RBF)",
     SVC(random_state=RANDOM_STATE), True),
    ("k-NN (k=5)",
     KNeighborsClassifier(n_neighbors=5), True),
    ("Gaussian Naive Bayes",
     GaussianNB(), False),
    ("Neural Net (MLP)",
     MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                   random_state=RANDOM_STATE), True),
]

def build_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    """Wrap an estimator in a Pipeline, with StandardScaler only if needed."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def bootstrap_accuracy_ci(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          n_resamples: int = N_BOOTSTRAP,
                          seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Return a 95% bootstrap confidence interval for test accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = accuracy_score(y_true[idx], y_pred[idx])

    return (
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )

def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("Classifier training and comparison — Steps 8-10")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found - run 04_extract_features.py first.")
    df = pd.read_csv(FEATURES_CSV)

    feature_cols = [c for c in df.columns
                    if c not in ("global_index", "satellite_id")]
    X = df[feature_cols].to_numpy()
    y = df["satellite_id"].to_numpy()
    class_labels = np.unique(y)
    print(f"Loaded {X.shape[0]:,} messages, {X.shape[1]} features, "
          f"{len(class_labels)} classes: {list(class_labels)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        stratify=y,                # keep class proportions in both sets
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(y_train):,} messages   Test: {len(y_test):,} messages")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    results = []
    fitted = {}
    reports_text = []

    for name, estimator, needs_scaling in MODELS:
        print(f"\n{name}")
        pipe = build_pipeline(estimator, needs_scaling)

        # Cross-validation ON TRAINING DATA ONLY (test set stays untouched)
        t0 = time.time()
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"  CV accuracy ({CV_FOLDS}-fold): "
              f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit on the full training set, evaluate once on held-out test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        ci_low, ci_high = bootstrap_accuracy_ci(y_test, y_pred)
        # Macro F1: F1 computed per class then averaged with EQUAL class
        # weight - so a model can't hide a badly-classified satellite
        # behind good performance on the others.
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        elapsed = time.time() - t0
        print(
            f"  Test accuracy: {acc:.4f} "
            f"(95% CI [{ci_low:.4f}, {ci_high:.4f}])   "
            f"Macro F1: {macro_f1:.4f}   ({elapsed:.1f} s)"
        )

        fitted[name] = pipe
        results.append({
            "model": name,
            "scaled": needs_scaling,
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "test_accuracy": round(acc, 4),
            "accuracy_ci_low": round(ci_low, 4),
            "accuracy_ci_high": round(ci_high, 4),
            "test_macro_f1": round(macro_f1, 4),
            "train_eval_seconds": round(elapsed, 1),
        })
        reports_text.append(
            f"{'=' * 60}\n{name}\n{'=' * 60}\n"
            + classification_report(y_test, y_pred, digits=4, zero_division=0)
        )

    comparison = pd.DataFrame(results).sort_values(
        "test_accuracy", ascending=False)
    comp_path = OUTPUT_TABLES / "classifier_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print("\n" + "=" * 70)
    print("Comparison (sorted by test accuracy):")
    print(comparison.to_string(index=False))

    rep_path = OUTPUT_REPORTS / "classification_reports.txt"
    rep_path.write_text("\n\n".join(reports_text))

    # Rows = true satellite, columns = predicted satellite. The diagonal
    # is correct classifications; off-diagonal cells show exactly WHICH
    # satellites get confused with each other - far more informative for
    # the dissertation than a single accuracy number.
    n_models = len(MODELS)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, _, _) in zip(axes, MODELS):
        cm = confusion_matrix(y_test, fitted[name].predict(X_test),
                              labels=class_labels)
        ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    for ax in axes[n_models:]:   # hide any unused grid cells
        ax.set_visible(False)
    fig.suptitle("Confusion matrices on the held-out test set "
                 f"({len(y_test):,} messages)", fontsize=13)
    fig.tight_layout()
    cm_path = OUTPUT_FIGURES / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Random Forest importance = mean decrease in Gini impurity: how much
    # each feature's splits improve class purity, averaged over all trees.
    # KNOWN CAVEAT (be ready to say this in the meeting): impurity
    # importance splits credit between correlated features - e.g. var_I
    # and std_I carry the same information (std = sqrt(var)), so their
    # individual scores understate their shared importance. The RANKING
    # of feature groups is still informative.
    rf = fitted["Random Forest"].named_steps["model"]
    importance = (pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
      .reset_index(drop=True))
    imp_path = OUTPUT_TABLES / "feature_importance.csv"
    importance.to_csv(imp_path, index=False)

    print("\nTop 10 features (Random Forest importance):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row.feature:<22s} {row.importance:.4f}")

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(importance.feature[::-1], importance.importance[::-1],
            color="steelblue")
    ax.set_xlabel("Random Forest feature importance (mean Gini decrease)")
    ax.set_title("All 28 features, ranked")
    fig.tight_layout()
    fi_path = OUTPUT_FIGURES / "feature_importance.png"
    fig.savefig(fi_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\nOutputs written:")
    for p in (comp_path, rep_path, cm_path, fi_path, imp_path):
        print(f"  {p.relative_to(PROJECT_ROOT)}")
    print(f"Total time: {time.time() - t_start:.1f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()
