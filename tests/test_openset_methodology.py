import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "21_openset_and_domain.py"
spec = importlib.util.spec_from_file_location("openset_domain", SCRIPT_PATH)
openset_domain = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(openset_domain)


def test_unknown_satellites_are_rejected_from_training():
    openset_domain.assert_identity_separation(
        np.array([51, 85, 92]), np.array([109])
    )

    try:
        openset_domain.assert_identity_separation(
            np.array([51, 85, 109]), np.array([109])
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("overlapping unknown identity was not rejected")


def test_pipeline_imputer_is_fitted_on_training_values_only():
    model = openset_domain.forest()
    train = np.array([[1.0, np.nan], [3.0, 10.0]])
    test = np.array([[100.0, np.nan]])

    model.fit(train, np.array([0, 1]))
    transformed = model.named_steps["imputer"].transform(test)

    assert model.named_steps["imputer"].statistics_.tolist() == [2.0, 10.0]
    assert transformed.tolist() == [[100.0, 10.0]]


def test_fixed_feature_selection_does_not_depend_on_test_labels():
    sets = {"v1": ["a"], "v2": ["b"]}
    all_features = ["a", "b"]

    assert openset_domain.fixed_feature_columns(sets, all_features) == all_features


def test_threshold_is_explicitly_threshold_free():
    assert "threshold-free" in openset_domain.THRESHOLD_PROVENANCE


def test_script_uses_canonical_pass_definition():
    from scripts.evaluation_common import PASS_GAP_SECONDS

    assert openset_domain.PASS_GAP_SECONDS == PASS_GAP_SECONDS
    assert not hasattr(openset_domain, "PASS_GAP_S")
