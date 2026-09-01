"""
fusion/01_evidence_adapter.py

Bridge from the real physical-layer experiment to the simulated higher-layer
fusion framework.

REAL EVIDENCE
-------------
- extracted RF features from real Iridium IQ captures
- Random Forest prediction
- RF confidence / max class probability
- physical true satellite label (for evaluation only)

SIMULATED DOWNSTREAM
--------------------
- claimed identity under attack scenarios
- HMAC credentials
- freshness values
- higher-layer attack scenarios

The Random Forest here is deliberately kept identical to the authoritative
classical baseline used in scripts/05_train_classifiers.py:
- 28 v1 waveform features
- stratified 80/20 split
- random_state = 42
- 200 trees
- no feature scaling (tree splits are scale invariant)

The output carries model provenance so every fusion result can be traced back
to the physical-layer experiment that produced it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from _shared import FUSION_TABLES_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
CLASSIFIER_COMPARISON_CSV = (
    PROJECT_ROOT / "outputs" / "tables" / "classifier_comparison.csv"
)

RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_TREES = 200
MODEL_NAME = "random_forest"
FEATURE_SET = "v1_handcrafted_28"
SPLIT_TYPE = "stratified_80_20"

NON_FEATURE = {
    "sample_id",
    "global_index",
    "index",
    "Unnamed: 0",
    "satellite_id",
}


def _check_against_primary_baseline(accuracy: float) -> None:
    """Compare with the saved primary RF baseline when that table exists."""
    if not CLASSIFIER_COMPARISON_CSV.exists():
        return

    comparison = pd.read_csv(CLASSIFIER_COMPARISON_CSV)
    if "model" not in comparison.columns or "test_accuracy" not in comparison.columns:
        return

    row = comparison[comparison["model"].astype(str) == "Random Forest"]
    if row.empty:
        return

    saved = float(row.iloc[0]["test_accuracy"])
    difference = abs(saved - accuracy)

    print(
        f"  Saved primary RF baseline: {saved:.4f} "
        f"(difference {difference:.6f})"
    )

    # The comparison CSV is rounded, so allow a small tolerance.
    if difference > 5e-4:
        raise AssertionError(
            "Fusion RF accuracy does not match the saved primary Random Forest "
            "baseline closely enough. Check the feature set, split and model "
            "configuration before continuing."
        )


def main() -> None:
    print("=" * 70)
    print("Evidence adapter - physical layer -> fusion evidence record")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Run the physical-layer pipeline first: {FEATURES_CSV}"
        )

    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]

    if len(feat_cols) != 28:
        raise ValueError(
            f"Expected 28 v1 waveform features, found {len(feat_cols)}: "
            f"{feat_cols}"
        )

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values found in features.csv. The authoritative "
            "baseline does not perform global pre-split imputation, so fix the "
            "feature table rather than silently changing preprocessing here."
        )

    print("\nTraining physical-layer model (Random Forest)...")

    idx = np.arange(len(y))
    itr, ite, ytr, yte = train_test_split(
        idx,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X[itr], ytr)

    proba = model.predict_proba(X[ite])
    classes = model.classes_

    pred = classes[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)
    accuracy = float((pred == yte).mean())

    print(f"  Test messages -> evidence records: {len(ite):,}")
    print(f"  Model accuracy on these: {accuracy:.4f}")
    _check_against_primary_baseline(accuracy)

    evidence = pd.DataFrame(
        {
            "message_id": np.arange(len(ite)),
            "source_row_index": ite,
            "true_satellite": yte,
            "claimed_satellite": yte,
            "rf_predicted_sat": pred,
            "rf_confidence": np.round(confidence, 6),
            "rf_matches_claim": pred == yte,
            "openset_score": np.round(confidence, 6),

            # Provenance fields
            "model_name": MODEL_NAME,
            "feature_set": FEATURE_SET,
            "n_features": len(feat_cols),
            "split_type": SPLIT_TYPE,
            "test_fraction": TEST_FRACTION,
            "random_seed": RANDOM_SEED,
            "rf_n_estimators": N_TREES,
            "rf_scaled": False,
            "rf_test_accuracy": round(accuracy, 6),
        }
    )

    out = FUSION_TABLES_DIR / "evidence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(out, index=False)

    print(
        f"\n  Confidence: min {confidence.min():.3f}, "
        f"median {np.median(confidence):.3f}, max {confidence.max():.3f}"
    )
    print(
        "  Records where prediction matches claim: "
        f"{evidence['rf_matches_claim'].mean():.1%}"
    )
    print("\nProvenance:")
    print(f"  model_name       = {MODEL_NAME}")
    print(f"  feature_set      = {FEATURE_SET}")
    print(f"  n_features       = {len(feat_cols)}")
    print(f"  split_type       = {SPLIT_TYPE}")
    print(f"  random_seed      = {RANDOM_SEED}")
    print(f"  rf_n_estimators  = {N_TREES}")
    print("  rf_scaled        = False")

    print(f"\nSaved: {out}")
    print("Next: fusion/02_higher_layer_sim.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
