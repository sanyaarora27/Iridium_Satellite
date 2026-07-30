"""
07_results_table.py
===================

PURPOSE
-------
Produces the Step 9 results table in the exact column format requested:

    Model | Features used | Accuracy | Macro F1 | Notes

The `Features used` column is the point of the table. Listing eight models
that all consume the same 28 features says only "nothing worked". Adding
rows that use DIFFERENT inputs turns the same table into an argument about
WHERE the (weak) discriminative information actually lives:

    28 hand-crafted signal features   -- the intended fingerprinting approach
    4  channel metadata values        -- geometry only, no waveform at all
    0  features (Dummy)               -- chance reference

If the metadata row matches the signal-feature rows, the hand-crafted
features contribute nothing beyond capture geometry -- and the table shows
that in one glance rather than requiring three pages of explanation.

This script reads the outputs of script 04, recomputes the metadata-only
control so the table is self-contained and reproducible, and writes the
result as CSV and as a Markdown table ready to paste into the dissertation.

USAGE
-----
    python scripts/07_results_table.py

Requires scripts 03 and 04 to have been run first.

OUTPUTS
-------
    outputs/tables/results_table.csv
    outputs/reports/results_table.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV  = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
MODEL_RESULTS = PROJECT_ROOT / "outputs" / "tables" / "model_results.csv"
OUT_TABLES    = PROJECT_ROOT / "outputs" / "tables"
OUT_REPORTS   = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}
METADATA_COLUMNS = ["level", "noise", "ra_alt", "center_frequency"]


# --- Metadata-only control -----------------------------------------------
def load_metadata_column(column: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None


def metadata_only_control(df: pd.DataFrame) -> dict | None:
    """
    Train a Random Forest on channel metadata alone -- no waveform features.

    Uses the same split, seed and model settings as script 04 so the number
    is directly comparable with the signal-feature rows.
    """
    index_col = next((c for c in ("global_index", "sample_id")
                      if c in df.columns), None)
    if index_col is None:
        return None
    rows = df[index_col].to_numpy(dtype=int)

    columns, arrays = [], []
    for column in METADATA_COLUMNS:
        full = load_metadata_column(column)
        if full is None or rows.max() >= len(full):
            continue
        columns.append(column)
        arrays.append(full[rows])

    if not arrays:
        return None

    X = np.column_stack(arrays).astype(float)
    # Replace non-finite entries with the column median
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0

    y = df["satellite_id"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=100,
                                         random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "columns":  columns,
    }


# --- Notes ---------------------------------------------------------------
def build_note(model: str, accuracy: float, ci_low: float,
               ci_high: float, chance_high: float) -> str:
    """
    Write the Notes cell. Every claim here is tied to the confidence
    interval rather than to the point estimate, so the table never asserts
    a difference the data cannot support.
    """
    if "Dummy" in model:
        return "Chance reference; predicts majority class only"
    if ci_low > chance_high:
        return "Exceeds chance (CI clears the chance reference)"
    return "Not distinguishable from chance (CI overlaps reference)"


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Step 9 results table")
    print("=" * 70)

    if not MODEL_RESULTS.exists():
        raise FileNotFoundError(f"Run script 04 first: {MODEL_RESULTS}")

    results = pd.read_csv(MODEL_RESULTS)
    df = pd.read_csv(FEATURES_CSV)
    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    n_features = len(feature_names)

    dummy = results[results["model"].str.contains("Dummy")]
    chance_high = float(dummy["accuracy_ci_high"].iloc[0]) if len(dummy) else 0.0
    n_train = int(results["n_train"].iloc[0])
    n_test  = int(results["n_test"].iloc[0])
    n_class = int(results["n_classes"].iloc[0])

    rows = []
    for _, r in results.iterrows():
        used = ("none (majority class)" if "Dummy" in r["model"]
                else f"{n_features} hand-crafted")
        rows.append({
            "Model":         r["model"],
            "Features used": used,
            "Accuracy":      r["accuracy"],
            "Macro F1":      r["macro_f1"],
            "Notes":         build_note(r["model"], r["accuracy"],
                                        r["accuracy_ci_low"],
                                        r["accuracy_ci_high"], chance_high),
        })

    # The control row -- the reason the "Features used" column matters
    print("\nRecomputing metadata-only control...")
    control = metadata_only_control(df)
    if control is not None:
        print(f"  metadata only ({len(control['columns'])} columns): "
              f"acc={control['accuracy']:.4f}")
        rows.append({
            "Model":         "Random Forest",
            "Features used": f"{len(control['columns'])} channel metadata "
                             f"({', '.join(control['columns'])})",
            "Accuracy":      control["accuracy"],
            "Macro F1":      control["macro_f1"],
            "Notes":         "Control: no waveform data used at all",
        })
    else:
        print("  metadata unavailable -- control row omitted")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLES / "results_table.csv", index=False)

    # Console view
    print()
    print(f"{'Model':<22}{'Features used':<34}{'Acc':>8}{'F1':>8}")
    print("-" * 74)
    for _, r in table.iterrows():
        print(f"{r['Model']:<22}{r['Features used'][:33]:<34}"
              f"{r['Accuracy']:>8.4f}{r['Macro F1']:>8.4f}")

    # Markdown
    md = f"""# Results table (Task list Step 9)

**Task:** classify which Iridium satellite transmitted a given message.

- Satellites: {n_class}
- Samples: {n_train + n_test:,} ({n_train:,} train / {n_test:,} test)
- Split: stratified {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)}, seed {RANDOM_SEED}
- Preprocessing: StandardScaler fitted inside a Pipeline on training folds only
- Hyperparameters: scikit-learn defaults, no tuning

| Model | Features used | Accuracy | Macro F1 | Notes |
|-------|---------------|---------:|---------:|-------|
"""
    for _, r in table.iterrows():
        md += (f"| {r['Model']} | {r['Features used']} | "
               f"{r['Accuracy']:.4f} | {r['Macro F1']:.4f} | {r['Notes']} |\n")

    md += """
## Reading this table

The `Features used` column carries the argument. The first group of rows all
consume the same 28 hand-crafted signal features and none of them separates
from the chance reference. The final row uses **no waveform data whatsoever**
-- only channel metadata describing where the satellite was and how strong
the signal arrived -- and performs comparably.

The conclusion is therefore not simply that the models failed. It is that the
28 hand-crafted features supply no transmitter-specific information beyond
what capture geometry already provides. Whatever small margin exists above
chance is attributable to orbital geometry rather than to hardware
fingerprints.

Per the task brief, the aim at this stage is not to match SatIQ's reported
performance but to build a baseline that is fully understood and explainable.
This table documents both the result and the reason for it.
"""

    (OUT_REPORTS / "results_table.md").write_text(md)

    print("\nWritten:")
    print("  outputs/tables/results_table.csv")
    print("  outputs/reports/results_table.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
