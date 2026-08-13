"""
01_evidence_adapter.py
======================

PURPOSE
-------
Convert the outputs of the physical-layer pipeline into a single per-message
evidence record that the higher-layer fusion logic can consume.

This is the bridge between the completed RF evaluation and the fusion
demonstration. It performs no new analysis and trains no model; it assembles
values already produced by earlier scripts into one table.

WHAT IS REAL AND WHAT IS SIMULATED
----------------------------------
The physical-layer evidence in this record is MEASURED from real Iridium
signals: the predicted satellite identity and the model's confidence come
from a Random Forest trained on the hand-crafted features, exactly as in the
baseline comparison.

The higher-layer fields (a message authentication code and a freshness
nonce) are NOT present in the Iridium data -- the protocol carries no such
values. They are added by the higher-layer simulator (script 02) to
represent a hypothetical authenticated protocol. This separation is
deliberate and is preserved in the field names: physical-layer fields are
named plainly, simulated fields are prefixed `sim_`. The dissertation must
describe the higher layer as a simulation for this reason.

EVIDENCE RECORD FIELDS
----------------------
    message_id            row identifier
    claimed_satellite     the identity the message asserts (see note below)
    rf_predicted_sat      satellite predicted by the physical-layer model
    rf_confidence         model's probability for its top class (0-1)
    rf_matches_claim      whether prediction equals the claim
    openset_score         max class probability; low implies "unknown"
    (higher-layer fields are added later by the simulator)

THE CLAIMED IDENTITY
--------------------
In an operational setting a message asserts an identity that the receiver
must verify. The dataset has no separate claim field, so under normal
(non-attack) conditions the claim is set equal to the true satellite. Attack
scenarios (script 03) then deliberately break this correspondence -- for
example by pairing one satellite's signal with another's claimed identity --
which is exactly what a spoofing or cloning attack does.

USAGE
-----
    python fusion/01_evidence_adapter.py

OUTPUT
------
    fusion/evidence.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
FUSION_DIR   = Path(__file__).resolve().parent
FUSION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
NON_FEATURE   = {"sample_id", "global_index", "index",
                 "Unnamed: 0", "satellite_id"}


def main() -> None:
    print("=" * 70)
    print("Evidence adapter - physical layer -> fusion evidence record")
    print("=" * 70)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Run the physical-layer pipeline first: "
                                f"{FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()

    # Train the physical-layer model and evaluate on a held-out split. Only
    # the test messages become evidence records, so every record carries a
    # genuine out-of-sample prediction rather than one the model has seen.
    print("\nTraining physical-layer model (Random Forest)...")
    idx = np.arange(len(y))
    itr, ite, ytr, yte = train_test_split(
        idx, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=300,
                                         random_state=RANDOM_SEED, n_jobs=-1)),
    ]).fit(X[itr], ytr)

    proba = model.predict_proba(X[ite])
    classes = model.named_steps["clf"].classes_
    pred = classes[np.argmax(proba, axis=1)]
    confidence = proba.max(axis=1)

    print(f"  Test messages -> evidence records: {len(ite):,}")
    print(f"  Model accuracy on these: {(pred == yte).mean():.4f}")

    # Build the evidence table. Under normal conditions the claimed identity
    # equals the true satellite; attack scenarios will override this later.
    evidence = pd.DataFrame({
        "message_id":        np.arange(len(ite)),
        "true_satellite":    yte,               # ground truth, for scoring only
        "claimed_satellite": yte,               # normal case: claim == truth
        "rf_predicted_sat":  pred,
        "rf_confidence":     np.round(confidence, 4),
        "rf_matches_claim":  (pred == yte),
        "openset_score":     np.round(confidence, 4),
    })

    out = FUSION_DIR / "evidence.csv"
    evidence.to_csv(out, index=False)

    print(f"\n  Confidence: min {confidence.min():.3f}, "
          f"median {np.median(confidence):.3f}, max {confidence.max():.3f}")
    print(f"  Records where prediction matches claim: "
          f"{evidence['rf_matches_claim'].mean():.1%}")
    print(f"\nSaved: {out}")
    print("Next: fusion/02_higher_layer_sim.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
