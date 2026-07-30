"""
07_train_classifiers_groupkfold.py
===================================

Purpose
-------
Re-evaluate the seven classifiers from 05_train_classifiers.py using GroupKFold
cross-validation, where the "group" is the satellite pass (a burst of messages
sharing similar channel conditions: elevation, Doppler, SNR/level).

Why this matters
----------------
Standard KFold splits messages randomly, so messages from the same pass can
appear in both train and test. A model can then "cheat" by memorising the
channel conditions of a pass rather than learning the transmitter's hardware
fingerprint. The Spearman rho = 0.997 between `level` metadata and extracted
`signal_power` (from 06_diagnose_features.py) confirms channel effects dominate
the feature space, so this leakage is not hypothetical.

GroupKFold enforces that all messages from the same pass stay together --
entirely in train or entirely in test -- so accuracy reflects generalisation
to *new passes*, which is what matters for RF fingerprinting.

Grouping logic
--------------
If a `pass_id` column exists in features.csv, use it directly.
Otherwise derive passes from timestamps: sort by (satellite_id, timestamp),
then start a new pass whenever the timestamp gap exceeds GAP_SECONDS or the
satellite_id changes. Iridium bursts are typically < 10 seconds apart within
a pass, so 60 s is a conservative threshold.

Outputs
-------
- outputs/tables/07_groupkfold_comparison.csv       (ranked mean accuracy)
- outputs/tables/07_groupkfold_per_fold.csv         (per-fold detail)
- outputs/tables/07_groupkfold_vs_kfold.csv         (comparison to script 05)
- outputs/figures/07_groupkfold_confusion_matrices.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Config -- adjust paths / column names here if they differ in your setup
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path.home() / "Desktop" / "Surrey" / "Iridium_Satellite"
FEATURES_CSV   = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
KFOLD_RESULTS  = PROJECT_ROOT / "outputs" / "tables" / "05_classifier_comparison.csv"  # from script 05, for comparison
OUT_TABLES     = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS       = PROJECT_ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

N_SPLITS       = 5
GAP_SECONDS    = 60.0                # gap > this starts a new pass
LABEL_COL      = "satellite_id"
RANDOM_STATE   = 42

# columns to exclude from feature matrix -- anything that isn't a feature
KNOWN_METADATA = {"satellite_id", "timestamp", "time", "epoch_time",
                  "level", "message_index", "msg_idx", "pass_id",
                  "file", "segment"}

# candidate timestamp column names to try, in order
TIMESTAMP_CANDIDATES = ["timestamp", "time", "epoch_time", "t"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading {FEATURES_CSV}")
df = pd.read_csv(FEATURES_CSV)
print(f"  {len(df)} rows, {len(df.columns)} columns")
print(f"  columns: {list(df.columns)}")

assert LABEL_COL in df.columns, (
    f"Expected label column '{LABEL_COL}' in features.csv. "
    f"Available: {list(df.columns)}"
)


# ---------------------------------------------------------------------------
# Build group ids (one integer per pass)
# ---------------------------------------------------------------------------
def build_groups(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return (groups, df_sorted).
    df_sorted has the same rows as df but reordered to match `groups`.
    """
    # Case 1: pre-existing pass_id -> use directly
    if "pass_id" in df.columns:
        print(f"Using existing 'pass_id' column ({df['pass_id'].nunique()} unique passes)")
        return df["pass_id"].values, df

    # Case 2: derive from timestamp
    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise KeyError(
            f"No timestamp column found (tried {TIMESTAMP_CANDIDATES}). "
            f"Add one to features.csv or set a pass_id column, then re-run. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Deriving passes from '{ts_col}' with gap threshold {GAP_SECONDS}s")
    df_s = df.sort_values([LABEL_COL, ts_col]).reset_index(drop=True)

    dt         = df_s[ts_col].diff()
    sat_change = df_s[LABEL_COL].ne(df_s[LABEL_COL].shift())
    new_pass   = (dt > GAP_SECONDS) | sat_change            # bool: does this row start a new pass?
    groups     = new_pass.cumsum().values                    # cumsum turns booleans into pass ids
    print(f"  {len(np.unique(groups))} passes derived")
    return groups, df_s


groups, df = build_groups(df)


# sanity: how many passes per class?
pass_per_class = (
    pd.DataFrame({"sat": df[LABEL_COL].values, "pass": groups})
      .drop_duplicates()
      .groupby("sat").size()
      .sort_values(ascending=False)
)
print("\nPasses per satellite:")
print(pass_per_class.to_string())

if pass_per_class.min() < N_SPLITS:
    print(f"\nWARNING: at least one satellite has fewer than {N_SPLITS} passes.")
    print("         GroupKFold may put all of that class's passes in a single fold.")


# ---------------------------------------------------------------------------
# Build X, y
# ---------------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in KNOWN_METADATA]
X = df[feature_cols].values.astype(float)
y = df[LABEL_COL].values
print(f"\nX shape: {X.shape}  ({len(feature_cols)} features)")
print(f"Classes: {np.unique(y).tolist()}")


# ---------------------------------------------------------------------------
# Models -- mirrors 05_train_classifiers.py exactly for direct comparison
# ---------------------------------------------------------------------------
def pipe(clf, scale: bool = True) -> Pipeline:
    """Wrap a classifier in a scikit-learn Pipeline, optionally with StandardScaler.

    Scaling is applied inside the Pipeline so it fits on train folds only,
    which prevents test-fold statistics leaking into training.
    """
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


models = {
    "Dummy":  pipe(DummyClassifier(strategy="most_frequent"), scale=False),
    "LogReg": pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "DT":     pipe(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False),
    "RF":     pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), scale=False),
    "SVM":    pipe(SVC(kernel="rbf", random_state=RANDOM_STATE)),
    "kNN":    pipe(KNeighborsClassifier(n_neighbors=5)),
    "NB":     pipe(GaussianNB(), scale=False),
}


# ---------------------------------------------------------------------------
# Run GroupKFold
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=N_SPLITS)
print(f"\nRunning GroupKFold(n_splits={N_SPLITS})\n" + "-" * 55)

summary_rows  = []
per_fold_rows = []
predictions   = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X, y, groups=groups, cv=gkf, scoring="accuracy", n_jobs=-1,
    )
    y_pred = cross_val_predict(
        model, X, y, groups=groups, cv=gkf, n_jobs=-1,
    )
    predictions[name] = y_pred

    summary_rows.append({
        "model":    name,
        "mean_acc": scores.mean(),
        "std_acc":  scores.std(),
        "min_fold": scores.min(),
        "max_fold": scores.max(),
    })
    for i, s in enumerate(scores):
        per_fold_rows.append({"model": name, "fold": i, "acc": s})

    print(f"{name:8s}  mean={scores.mean():.3f}  std={scores.std():.3f}  "
          f"folds={np.round(scores, 3).tolist()}")


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------
comparison = pd.DataFrame(summary_rows).sort_values("mean_acc", ascending=False)
comparison.to_csv(OUT_TABLES / "07_groupkfold_comparison.csv", index=False)

pd.DataFrame(per_fold_rows).to_csv(OUT_TABLES / "07_groupkfold_per_fold.csv", index=False)


# side-by-side comparison to 05 (leaky KFold) if that file exists
if KFOLD_RESULTS.exists():
    kfold = pd.read_csv(KFOLD_RESULTS)
    # try to align on model name -- adjust column names if 05 saved them differently
    model_col = "model" if "model" in kfold.columns else kfold.columns[0]
    acc_col   = next((c for c in kfold.columns if "acc" in c.lower() and "mean" in c.lower()), None) \
                or next((c for c in kfold.columns if "acc" in c.lower()), None)
    if acc_col is not None:
        merged = comparison.merge(
            kfold[[model_col, acc_col]].rename(columns={model_col: "model", acc_col: "kfold_acc"}),
            on="model", how="left",
        )
        merged["delta"] = merged["mean_acc"] - merged["kfold_acc"]
        merged.to_csv(OUT_TABLES / "07_groupkfold_vs_kfold.csv", index=False)
        print(f"\nSide-by-side vs script 05:")
        print(merged.to_string(index=False))
    else:
        print(f"\nCould not find accuracy column in {KFOLD_RESULTS.name}; skipping comparison.")
else:
    print(f"\n({KFOLD_RESULTS.name} not found -- skipping side-by-side vs script 05.)")


# ---------------------------------------------------------------------------
# Confusion matrices (row-normalised)
# ---------------------------------------------------------------------------
classes = np.unique(y)
n_models = len(models)
n_cols = 3
n_rows = int(np.ceil(n_models / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
axes = np.atleast_1d(axes).flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(name)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")

for ax in axes[n_models:]:
    ax.axis("off")

fig.suptitle("GroupKFold Confusion Matrices (row-normalised)", y=1.01)
fig.tight_layout()
fig.savefig(OUT_FIGS / "07_groupkfold_confusion_matrices.png", dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
print("\n=== Ranked by GroupKFold mean accuracy ===")
print(comparison.to_string(index=False))

chance = 1.0 / len(classes)
print(f"\nChance level for {len(classes)} balanced classes: {chance:.3f}")
print(f"Best model: {comparison.iloc[0]['model']} "
      f"({comparison.iloc[0]['mean_acc']:.3f} +/- {comparison.iloc[0]['std_acc']:.3f})")

print(f"\nSaved:")
print(f"  {OUT_TABLES / '07_groupkfold_comparison.csv'}")
print(f"  {OUT_TABLES / '07_groupkfold_per_fold.csv'}")
if (OUT_TABLES / "07_groupkfold_vs_kfold.csv").exists():
    print(f"  {OUT_TABLES / '07_groupkfold_vs_kfold.csv'}")
print(f"  {OUT_FIGS / '07_groupkfold_confusion_matrices.png'}")
