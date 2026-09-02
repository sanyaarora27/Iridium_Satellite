"""Shared definitions for classical physical-layer evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd

PASS_GAP_SECONDS = 300

def timestamp_global_to_seconds(timestamp_global: np.ndarray) -> np.ndarray:
    """Convert the dataset's nanosecond timestamp_global values to seconds."""
    return np.asarray(timestamp_global, dtype=np.float64) / 1e9

def assign_inferred_passes(
    dataframe: pd.DataFrame,
    satellite_column: str = "satellite_id",
    timestamp_column: str = "timestamp_global",
    index_column: str = "global_index",
    gap_seconds: float = PASS_GAP_SECONDS,
) -> np.ndarray:
    """Assign deterministic timestamp-gap inferred passes to original rows.

    Messages are grouped per satellite, ordered by timestamp and then by
    global_index, and split when the chronological gap exceeds gap_seconds.
    The returned IDs are globally unique and aligned with dataframe rows.
    This is an operational grouping heuristic, not an ephemeris-derived orbit.
    """
    required = {satellite_column, timestamp_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise KeyError(f"Missing inferred-pass column(s): {sorted(missing)}")
    if index_column not in dataframe.columns:
        order_index = np.arange(len(dataframe), dtype=np.int64)
    else:
        order_index = dataframe[index_column].to_numpy(dtype=np.int64)

    timestamps = timestamp_global_to_seconds(dataframe[timestamp_column].to_numpy())
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamp_global contains non-finite values")

    pass_ids = np.empty(len(dataframe), dtype=np.int64)
    next_pass_id = 0
    satellites = dataframe[satellite_column].to_numpy()

    for satellite in np.sort(pd.unique(satellites)):
        row_indices = np.flatnonzero(satellites == satellite)
        sort_order = np.lexsort((order_index[row_indices], timestamps[row_indices]))
        ordered_rows = row_indices[sort_order]
        pass_ids[ordered_rows[0]] = next_pass_id

        for previous_row, current_row in zip(ordered_rows[:-1], ordered_rows[1:]):
            if timestamps[current_row] - timestamps[previous_row] > gap_seconds:
                next_pass_id += 1
            pass_ids[current_row] = next_pass_id
        next_pass_id += 1

    return pass_ids