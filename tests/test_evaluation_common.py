import pandas as pd

from scripts.evaluation_common import PASS_GAP_SECONDS, assign_inferred_passes


def test_pass_ids_are_unique_across_satellites():
    df = pd.DataFrame({
        "satellite_id": [51, 51, 85, 85],
        "timestamp_global": [0, 1, 0, 1],
        "global_index": [0, 1, 2, 3],
    })

    pass_ids = assign_inferred_passes(df)

    assert len(set(pass_ids[df.satellite_id == 51])) == 1
    assert len(set(pass_ids[df.satellite_id == 85])) == 1
    assert set(pass_ids[df.satellite_id == 51]).isdisjoint(
        set(pass_ids[df.satellite_id == 85])
    )


def test_gap_at_or_below_threshold_stays_in_same_pass():
    df = pd.DataFrame({
        "satellite_id": [51, 51],
        "timestamp_global": [0, PASS_GAP_SECONDS * 1e9],
        "global_index": [0, 1],
    })

    pass_ids = assign_inferred_passes(df)

    assert pass_ids[0] == pass_ids[1]


def test_gap_above_threshold_starts_new_pass():
    df = pd.DataFrame({
        "satellite_id": [51, 51],
        "timestamp_global": [0, (PASS_GAP_SECONDS + 1) * 1e9],
        "global_index": [0, 1],
    })

    pass_ids = assign_inferred_passes(df)

    assert pass_ids[0] != pass_ids[1]


def test_row_order_does_not_change_original_row_assignments():
    df = pd.DataFrame({
        "satellite_id": [51, 85, 51, 85],
        "timestamp_global": [2e9, 1e9, 1e9, 2e9],
        "global_index": [2, 1, 0, 3],
    })

    original_ids = assign_inferred_passes(df)
    shuffled = df.sample(frac=1, random_state=7)
    shuffled_ids = assign_inferred_passes(shuffled)
    restored = pd.Series(shuffled_ids, index=shuffled.index).sort_index().to_numpy()

    assert (original_ids == restored).all()


def test_equal_timestamps_use_global_index_as_tie_breaker():
    df = pd.DataFrame({
        "satellite_id": [51, 51, 51],
        "timestamp_global": [0, 0, 301e9],
        "global_index": [10, 2, 11],
    })

    pass_ids = assign_inferred_passes(df)

    assert pass_ids[0] == pass_ids[1]
    assert pass_ids[2] != pass_ids[0]
