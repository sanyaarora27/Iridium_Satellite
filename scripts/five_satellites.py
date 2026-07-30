"""
five_satellites.py

We chose the 5 most-sampled satellite IDs from the dataset:
    92, 85, 87, 51, 109
Each has 1,180-1,282 messages, so class balance is good.

For each satellite, we produce 5 diagnostic views:
    1. I (in-phase / real part) over time
    2. Q (quadrature / imaginary part) over time
    3. Amplitude = sqrt(I^2 + Q^2)
    4. Phase = arctan2(Q, I)
    5. FFT magnitude spectrum

The result is a 5-row x 5-column grid figure that lets you visually
compare all satellites.

WHY WE DO IT THIS WAY
---------------------
The dataset stores IQ samples across 5 segment files:
    samples_000.npy, samples_001.npy, ..., samples_004.npy
Each is ~880 MB (10,000 messages of 11,000 samples of I+Q float32).

To avoid loading everything (4.4 GB in RAM), we:
  1. First load only the small ra_sat_*.npy files (all satellite IDs).
  2. For each target satellite, find the FIRST message belonging to it
     and note which segment file it lives in.
  3. Load only the segment files we actually need.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---------------------------------------------------------------
# The satellites we chose (top 5 most-sampled in the dataset)
TARGET_SATELLITES = [92, 85, 87, 51, 109]

# Sampling rate used by the Watch This Space / SatIQ receivers
SAMPLE_RATE_HZ = 25_000_000

# Number of messages per segment file
MESSAGES_PER_SEGMENT = 10000


# --- STEP 1: LOAD ALL SATELLITE ID FILES ----------------------------------
# ra_sat_*.npy files are tiny (each holds 10,000 integers = ~80 KB).
# Loading all 5 costs about 400 KB of RAM. Easy.
sat_id_files = sorted(DATA_DIR.glob("ra_sat_*.npy"))
print(f"Found {len(sat_id_files)} satellite-ID files")

# Build a flat list of all satellite IDs across all segments.
# The result is a 1D array of length 50,000 where element i is the
# satellite ID of the i-th message overall (concatenated across segments).
all_satellite_ids = np.concatenate([np.load(f) for f in sat_id_files])
print(f"Total messages available: {len(all_satellite_ids):,}")


# --- STEP 2: FOR EACH TARGET SATELLITE, PICK ONE MESSAGE ------------------
# We want ONE representative message per satellite. We pick the first
# one we find. This is a defensible choice: no cherry-picking.
#
# For each target satellite, we record:
#   - global_index: the message's position in the concatenated 50K array
#   - segment_number: which samples_NNN.npy file it lives in (0-4)
#   - local_index: its row within that segment file (0-9999)

def locate_first_message(satellite_id: int) -> dict:
    """Find the first message index in the flat 50K array for a satellite."""
    matching_indices = np.where(all_satellite_ids == satellite_id)[0]
    if len(matching_indices) == 0:
        raise ValueError(f"Satellite {satellite_id} not found in dataset")

    global_index = int(matching_indices[0])
    # Which segment file? Which row within it?
    #   segment 0 holds messages 0-9999
    #   segment 1 holds messages 10000-19999
    #   etc.
    segment_number = global_index // MESSAGES_PER_SEGMENT
    local_index    = global_index %  MESSAGES_PER_SEGMENT
    return {
        "satellite_id":   satellite_id,
        "global_index":   global_index,
        "segment_number": segment_number,
        "local_index":    local_index,
    }

# Build a list of location records
message_locations = [locate_first_message(sat) for sat in TARGET_SATELLITES]

# Print a summary of what we're loading
print("\nSelected messages:")
for loc in message_locations:
    print(f"  Sat {loc['satellite_id']:>3d}: "
          f"segment {loc['segment_number']}, "
          f"row {loc['local_index']} "
          f"(global index {loc['global_index']:,})")


# --- STEP 3: LOAD ONLY THE SEGMENTS WE NEED -------------------------------
# Some satellites might live in the same segment file - if so, we should
# load that file only once. Use a dictionary to cache loaded segments.

segment_cache: dict[int, np.ndarray] = {}

def get_message_iq(location: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the (I, Q) arrays for the message described by 'location'."""
    seg_num = location["segment_number"]
    if seg_num not in segment_cache:
        seg_path = DATA_DIR / f"samples_{seg_num:03d}.npy"
        print(f"  Loading {seg_path.name}...", end=" ", flush=True)
        segment_cache[seg_num] = np.load(seg_path)
        print("done")

    segment = segment_cache[seg_num]
    # One message: shape (11000, 2) with [I, Q] in the last axis
    message = segment[location["local_index"]]
    i_values = message[:, 0]
    q_values = message[:, 1]
    return i_values, q_values


# --- STEP 4: COMPUTE DERIVED QUANTITIES FOR EACH MESSAGE ------------------
# For each of the 5 satellites, compute all 5 diagnostic views.

def compute_all_views(i_values: np.ndarray, q_values: np.ndarray) -> dict:
    """Compute the 5 things we're going to plot for one message."""
    n_samples = len(i_values)
    time_us = np.arange(n_samples) / SAMPLE_RATE_HZ * 1e6  # microseconds

    amplitude = np.sqrt(i_values**2 + q_values**2)
    phase     = np.arctan2(q_values, i_values)

    # FFT of the complex signal
    complex_signal = i_values + 1j * q_values
    fft_shifted    = np.fft.fftshift(np.fft.fft(complex_signal))
    fft_magnitude  = np.abs(fft_shifted)
    fft_freqs_mhz  = np.fft.fftshift(
        np.fft.fftfreq(n_samples, d=1.0 / SAMPLE_RATE_HZ)
    ) / 1e6

    return {
        "time_us":       time_us,
        "i_values":      i_values,
        "q_values":      q_values,
        "amplitude":     amplitude,
        "phase":         phase,
        "fft_freqs_mhz": fft_freqs_mhz,
        "fft_magnitude": fft_magnitude,
    }


# Load and compute for each satellite
print("\nLoading and processing messages:")
computed_views = []
for loc in message_locations:
    print(f"  Sat {loc['satellite_id']}: ", end="")
    i_vals, q_vals = get_message_iq(loc)
    views = compute_all_views(i_vals, q_vals)
    views["satellite_id"] = loc["satellite_id"]
    computed_views.append(views)


# --- STEP 5: BUILD THE 5x5 GRID FIGURE ------------------------------------
# Rows are satellites (5 of them), columns are the 5 views.
# The result is a 5x5 grid of subplots.

n_satellites = len(TARGET_SATELLITES)
n_views      = 5
fig, axes = plt.subplots(
    n_satellites, n_views,
    figsize=(20, 3.2 * n_satellites),
    sharex="col",
)

# Colours for the 5 view types (consistent across satellites)
view_colours = {
    "I":         "steelblue",
    "Q":         "darkorange",
    "Amplitude": "seagreen",
    "Phase":     "purple",
    "FFT":       "crimson",
}

for row_idx, views in enumerate(computed_views):
    sat_id = views["satellite_id"]

    # Column 0: I over time
    ax = axes[row_idx, 0]
    ax.plot(views["time_us"], views["i_values"],
            color=view_colours["I"], linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.set_ylabel(f"Sat {sat_id}", fontweight="bold")
    if row_idx == 0:
        ax.set_title("I (in-phase)")

    # Column 1: Q over time
    ax = axes[row_idx, 1]
    ax.plot(views["time_us"], views["q_values"],
            color=view_colours["Q"], linewidth=0.5)
    ax.grid(alpha=0.3)
    if row_idx == 0:
        ax.set_title("Q (quadrature)")

    # Column 2: Amplitude
    ax = axes[row_idx, 2]
    ax.plot(views["time_us"], views["amplitude"],
            color=view_colours["Amplitude"], linewidth=0.5)
    ax.grid(alpha=0.3)
    if row_idx == 0:
        ax.set_title("Amplitude")

    # Column 3: Phase
    ax = axes[row_idx, 3]
    ax.plot(views["time_us"], views["phase"],
            color=view_colours["Phase"], linewidth=0.5)
    ax.grid(alpha=0.3)
    if row_idx == 0:
        ax.set_title("Phase (rad)")

    # Column 4: FFT
    ax = axes[row_idx, 4]
    ax.plot(views["fft_freqs_mhz"], views["fft_magnitude"],
            color=view_colours["FFT"], linewidth=0.5)
    ax.grid(alpha=0.3)
    if row_idx == 0:
        ax.set_title("FFT magnitude")

# Bottom-row X labels
axes[-1, 0].set_xlabel("Time (us)")
axes[-1, 1].set_xlabel("Time (us)")
axes[-1, 2].set_xlabel("Time (us)")
axes[-1, 3].set_xlabel("Time (us)")
axes[-1, 4].set_xlabel("Freq offset (MHz)")

# Overall title
fig.suptitle(
    "Comparison of one IQ message from each of 5 Iridium satellites",
    fontsize=14, fontweight="bold", y=1.00
)

plt.tight_layout()

# Save the figure to disk
output_path = OUTPUT_DIR / "five_satellites_comparison.png"
plt.savefig(output_path, dpi=110, bbox_inches="tight")
print(f"\nSaved figure: {output_path}")

# Show interactively
plt.show()
