"""
Pass-aware evaluation:
“Does the model still recognise satellites during a new pass?”
===========================

Purpose
-------
The pass-aware script tests whether the classifiers can recognise the same satellites under different satellite-pass conditions.

A satellite pass means one continuous period during which a satellite is visible to the receiver. Messages from the same pass can share similar:

signal strength; Doppler shift; elevation; noise; propagation conditions.

Approach-> 
    1. Load feature.csv
    2. Load timestamp(global_index)
    3. Convert timestamps to seconds(nanoseconds to seconds)
    4. pass_id (It creates a new pass_id when: the satellite changes; or the time gap between consecutive messages exceeds your selected threshold.)
    
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
    cross_val_predict,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# StratifiedGroupKFold is available in recent scikit-learn versions.
# Fall back to GroupKFold if the installed version does not provide it.
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False


# ─── PATHS ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

for directory in (OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_REPORTS):
    directory.mkdir(parents=True, exist_ok=True)


# ─── CONFIG ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS = 5

# A new pass begins when the gap between consecutive messages from the same
# satellite is greater than this threshold.
PASS_GAP_SECONDS = 300.0

LABEL_COLUMN = "satellite_id"
INDEX_COLUMN = "global_index"

NON_FEATURE_COLUMNS = {
    "satellite_id",
    "global_index",
    "sample_id",
    "index",
    "Unnamed: 0",
    "timestamp",
    "timestamp_global",
    "pass_id",
}

MODELS = [
    (
        "Baseline (most frequent)",
        DummyClassifier(strategy="most_frequent"),
        False,
    ),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        True,
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        False,
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    (
        "SVM (RBF)",
        SVC(kernel="rbf", random_state=RANDOM_STATE),
        True,
    ),
    (
        "k-NN (k=5)",
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        True,
    ),
    (
        "Gaussian Naive Bayes",
        GaussianNB(),
        False,
    ),
    (
        "Neural Net (MLP)",
        MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        True,
    ),
]


def build_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    """Create a leakage-safe preprocessing and classifier pipeline."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def load_segmented_column(prefix: str) -> np.ndarray:
    """Concatenate all numbered .npy segments for one metadata column."""
    files = sorted(DATA_DIR.glob(f"{prefix}_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{prefix}_*.npy' were found in {DATA_DIR}"
        )

    arrays = [np.load(path, mmap_mode="r") for path in files]
    return np.concatenate(arrays)


def convert_timestamp_to_seconds(timestamp: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Convert timestamps to seconds using a conservative automatic unit check.

    The absolute timestamp size is less useful than the positive differences
    between consecutive values, so the median positive difference is inspected.
    """
    ts = np.asarray(timestamp, dtype=np.float64)

    sorted_ts = np.sort(ts[np.isfinite(ts)])
    positive_diffs = np.diff(sorted_ts)
    positive_diffs = positive_diffs[positive_diffs > 0]

    if len(positive_diffs) == 0:
        return ts, "seconds (no unit conversion inferred)"

    median_gap = float(np.median(positive_diffs))

    # Typical recording gaps should be fractions of a second to several seconds.
    if median_gap > 1e8:
        return ts / 1e9, "nanoseconds converted to seconds"
    if median_gap > 1e5:
        return ts / 1e6, "microseconds converted to seconds"
    if median_gap > 1e2:
        return ts / 1e3, "milliseconds converted to seconds"

    return ts, "seconds"


def attach_timestamps(features: pd.DataFrame) -> pd.DataFrame:
    """
    Attach raw timestamps to features.csv by using global_index.

    timestamp_global is preferred because it is continuous across data segments.
    timestamp is used only as a fallback.
    """
    if "timestamp_global" in features.columns:
        print("Using timestamp_global already present in features.csv")
        return features

    if INDEX_COLUMN not in features.columns:
        raise KeyError(
            f"'{INDEX_COLUMN}' is required to align feature rows with raw timestamps."
        )

    timestamp_prefix = None
    if list(DATA_DIR.glob("timestamp_global_*.npy")):
        timestamp_prefix = "timestamp_global"
    elif list(DATA_DIR.glob("timestamp_*.npy")):
        timestamp_prefix = "timestamp"

    if timestamp_prefix is None:
        raise FileNotFoundError(
            "No timestamp_global_*.npy or timestamp_*.npy files were found. "
            "Pass-aware grouping requires timestamp metadata."
        )

    print(f"Loading raw '{timestamp_prefix}' metadata...")
    all_timestamps = load_segmented_column(timestamp_prefix)

    global_indices = features[INDEX_COLUMN].to_numpy(dtype=np.int64)
    if global_indices.min() < 0 or global_indices.max() >= len(all_timestamps):
        raise IndexError(
            "A global_index value falls outside the raw timestamp array. "
            "Check that features.csv came from this exact dataset."
        )

    features = features.copy()
    features["timestamp_global"] = all_timestamps[global_indices]
    return features


def create_pass_ids(
    dataframe: pd.DataFrame,
    gap_seconds: float,
) -> pd.DataFrame:
    """
    Infer passes separately for each satellite.

    After sorting messages by satellite and time, a new pass starts whenever:
      1. the satellite changes, or
      2. the time gap exceeds gap_seconds.

    The returned pass_id is globally unique.
    """
    df = dataframe.copy()
    timestamp_seconds, unit_note = convert_timestamp_to_seconds(
        df["timestamp_global"].to_numpy()
    )
    df["timestamp_seconds"] = timestamp_seconds

    df = df.sort_values(
        [LABEL_COLUMN, "timestamp_seconds", INDEX_COLUMN]
    ).reset_index(drop=True)

    satellite_changed = df[LABEL_COLUMN].ne(df[LABEL_COLUMN].shift())
    time_gap = df.groupby(LABEL_COLUMN)["timestamp_seconds"].diff()
    large_gap = time_gap.gt(gap_seconds)

    df["pass_id"] = (satellite_changed | large_gap).cumsum().astype(np.int64)

    print(f"Timestamp interpretation: {unit_note}")
    print(f"Pass gap threshold: {gap_seconds:.1f} seconds")
    print(f"Derived passes: {df['pass_id'].nunique()}")

    return df


def validate_passes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise and validate the number and size of inferred passes."""
    summary = (
        df.groupby([LABEL_COLUMN, "pass_id"], as_index=False)
        .agg(
            message_count=(INDEX_COLUMN, "size"),
            pass_start=("timestamp_seconds", "min"),
            pass_end=("timestamp_seconds", "max"),
        )
    )
    summary["duration_seconds"] = summary["pass_end"] - summary["pass_start"]

    passes_per_satellite = (
        summary.groupby(LABEL_COLUMN)["pass_id"]
        .nunique()
        .sort_index()
    )

    print("\nPasses per satellite:")
    for satellite_id, count in passes_per_satellite.items():
        print(f"  Satellite {int(satellite_id):>3d}: {int(count):>4d} passes")

    minimum_passes = int(passes_per_satellite.min())
    if minimum_passes < 2:
        raise ValueError(
            "At least one satellite has fewer than two inferred passes. "
            "Pass-aware evaluation is not possible with the current grouping."
        )

    if minimum_passes < N_SPLITS:
        print(
            f"\nWarning: the smallest class has only {minimum_passes} passes. "
            f"The number of folds will be reduced from {N_SPLITS} "
            f"to {minimum_passes}."
        )

    return summary


def make_splitter(n_splits: int):
    """Prefer class-balanced grouped folds while keeping passes disjoint."""
    if HAS_STRATIFIED_GROUP_KFOLD:
        print(f"\nUsing StratifiedGroupKFold with {n_splits} folds")
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    print(
        f"\nStratifiedGroupKFold is unavailable; using GroupKFold "
        f"with {n_splits} folds"
    )
    return GroupKFold(n_splits=n_splits)


def check_fold_integrity(splitter, X, y, groups):
    """Create splits once and verify that no pass appears on both sides."""
    splits = list(splitter.split(X, y, groups))

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)

        if overlap:
            raise RuntimeError(
                f"Fold {fold_number} contains {len(overlap)} passes in both "
                "training and testing."
            )

        train_classes = set(y[train_idx])
        test_classes = set(y[test_idx])
        all_classes = set(np.unique(y))

        missing_train = all_classes - train_classes
        missing_test = all_classes - test_classes

        print(
            f"  Fold {fold_number}: train={len(train_idx):,}, "
            f"test={len(test_idx):,}, "
            f"train passes={len(train_groups)}, "
            f"test passes={len(test_groups)}"
        )

        # Add this block here
        print("    Test class distribution:")

        unique_labels, label_counts = np.unique(
            y[test_idx],
            return_counts=True
        )

        for label, count in zip(unique_labels, label_counts):
            print(f"      Satellite {int(label)}: {int(count)}")

        if missing_train:
            print(f"    Warning: missing training classes {sorted(missing_train)}")

        if missing_test:
            print(f"    Warning: missing test classes {sorted(missing_test)}")

    return splits


def plot_confusion_matrices(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    class_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save one row-normalised confusion matrix per model."""
    n_models = len(predictions)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrix_normalised = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )

        ConfusionMatrixDisplay(
            confusion_matrix=matrix_normalised,
            display_labels=class_labels,
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted satellite")
        ax.set_ylabel("True satellite")

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        "Pass-aware cross-validated confusion matrices",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: pd.DataFrame,
    pass_summary: pd.DataFrame,
    n_messages: int,
    n_features: int,
    n_splits: int,
    output_path: Path,
) -> None:
    """Write a concise dissertation-ready Markdown report."""
    best = results.iloc[0]
    n_passes = int(pass_summary["pass_id"].nunique())
    n_satellites = int(pass_summary[LABEL_COLUMN].nunique())

    report = f"""# Pass-aware classifier evaluation

## Purpose

This experiment evaluates whether the classifiers can identify a satellite from
messages belonging to previously unseen satellite passes. All messages assigned
to one pass remain entirely within either the training fold or the test fold.

## Experimental setup

- **Messages:** {n_messages:,}
- **Satellites:** {n_satellites}
- **RF features:** {n_features}
- **Inferred passes:** {n_passes}
- **Pass definition:** a new pass begins after a time gap greater than {PASS_GAP_SECONDS:.0f} seconds
- **Evaluation:** {n_splits}-fold grouped cross-validation
- **Leakage control:** no pass appears in both training and test data within a fold

## Results

| Model | Mean accuracy | Accuracy SD | Mean macro-F1 | Minimum fold | Maximum fold |
|---|---:|---:|---:|---:|---:|
"""

    for _, row in results.iterrows():
        report += (
            f"| {row['model']} "
            f"| {row['mean_accuracy']:.4f} "
            f"| {row['std_accuracy']:.4f} "
            f"| {row['mean_macro_f1']:.4f} "
            f"| {row['min_fold_accuracy']:.4f} "
            f"| {row['max_fold_accuracy']:.4f} |\n"
        )

    report += f"""

## Interpretation

The strongest pass-aware model was **{best['model']}**, with mean accuracy
**{best['mean_accuracy']:.1%}** and mean macro-F1
**{best['mean_macro_f1']:.3f}**.

Pass-aware performance should be compared with the random message-level split.
A substantial reduction indicates that the random split benefited from
pass-specific channel conditions rather than a stable transmitter fingerprint.
"""

    output_path.write_text(report)


def main() -> None:
    start_time = time.time()

    print("=" * 76)
    print("Pass-aware satellite classifier evaluation")
    print("=" * 76)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} does not exist. Run feature extraction first."
        )

    print(f"Loading {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)

    required_columns = {LABEL_COLUMN, INDEX_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"features.csv is missing required columns: {sorted(missing_columns)}"
        )

    df = attach_timestamps(df)
    df = create_pass_ids(df, PASS_GAP_SECONDS)
    pass_summary = validate_passes(df)

    pass_summary_path = OUTPUT_TABLES / "pass_summary.csv"
    pass_summary.to_csv(pass_summary_path, index=False)

    feature_columns = [
        column for column in df.columns
        if column not in NON_FEATURE_COLUMNS
        and column != "timestamp_seconds"
    ]

    if not feature_columns:
        raise ValueError("No model feature columns were found.")

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df[LABEL_COLUMN].to_numpy()
    groups = df["pass_id"].to_numpy()
    class_labels = np.unique(y)

    if not np.isfinite(X).all():
        bad_count = int((~np.isfinite(X)).sum())
        raise ValueError(
            f"The feature matrix contains {bad_count} NaN or infinite values. "
            "Clean features.csv before evaluation."
        )

    passes_per_class = (
        pass_summary.groupby(LABEL_COLUMN)["pass_id"].nunique()
    )
    n_splits = min(N_SPLITS, int(passes_per_class.min()))

    splitter = make_splitter(n_splits)
    print("\nChecking fold integrity:")
    splits = check_fold_integrity(splitter, X, y, groups)

    summary_rows = []
    per_fold_rows = []
    predictions = {}

    print("\nTraining models:")
    for model_name, estimator, needs_scaling in MODELS:
        print(f"\n{model_name}")
        pipeline = build_pipeline(estimator, needs_scaling)

        fold_accuracies = []
        fold_macro_f1 = []

        # cross_val_predict produces one prediction for every message using
        # a model that did not train on that message's entire pass.
        y_pred = cross_val_predict(
            pipeline,
            X,
            y,
            groups=groups,
            cv=splits,
            n_jobs=-1,
            method="predict",
        )
        predictions[model_name] = y_pred

        for fold_number, (_, test_idx) in enumerate(splits, start=1):
            fold_accuracy = accuracy_score(y[test_idx], y_pred[test_idx])
            fold_f1 = f1_score(
                y[test_idx],
                y_pred[test_idx],
                average="macro",
                zero_division=0,
            )
            fold_accuracies.append(fold_accuracy)
            fold_macro_f1.append(fold_f1)

            per_fold_rows.append({
                "model": model_name,
                "fold": fold_number,
                "test_messages": len(test_idx),
                "test_passes": len(np.unique(groups[test_idx])),
                "accuracy": fold_accuracy,
                "macro_f1": fold_f1,
            })

        summary_rows.append({
            "model": model_name,
            "scaled": needs_scaling,
            "mean_accuracy": np.mean(fold_accuracies),
            "std_accuracy": np.std(fold_accuracies),
            "mean_macro_f1": np.mean(fold_macro_f1),
            "std_macro_f1": np.std(fold_macro_f1),
            "min_fold_accuracy": np.min(fold_accuracies),
            "max_fold_accuracy": np.max(fold_accuracies),
        })

        print(
            f"  Accuracy: {np.mean(fold_accuracies):.4f} "
            f"+/- {np.std(fold_accuracies):.4f}"
        )
        print(
            f"  Macro-F1: {np.mean(fold_macro_f1):.4f} "
            f"+/- {np.std(fold_macro_f1):.4f}"
        )

    results = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    per_fold = pd.DataFrame(per_fold_rows)

    results_path = OUTPUT_TABLES / "pass_aware_results.csv"
    per_fold_path = OUTPUT_TABLES / "pass_aware_per_fold.csv"
    figure_path = OUTPUT_FIGURES / "pass_aware_confusion_matrices.png"
    report_path = OUTPUT_REPORTS / "pass_aware_evaluation.md"

    results.to_csv(results_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)

    plot_confusion_matrices(
        predictions,
        y,
        class_labels,
        figure_path,
    )

    write_report(
        results,
        pass_summary,
        n_messages=len(df),
        n_features=len(feature_columns),
        n_splits=n_splits,
        output_path=report_path,
    )

    print("\n" + "=" * 76)
    print("Pass-aware results ranked by mean accuracy:")
    print(results.to_string(index=False))
    print("\nOutputs written:")
    for path in (
        results_path,
        per_fold_path,
        pass_summary_path,
        figure_path,
        report_path,
    ):
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
    print("=" * 76)


if __name__ == "__main__":
    main()
