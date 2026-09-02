"""
iq_utils.py
===========

Shared signal-processing definitions for the Iridium classical-ML pipeline.

WHY THIS FILE EXISTS
--------------------
Amplitude, phase, and the FFT spectrum are needed by three scripts:
the single-signal plot, the five-satellite comparison plot, and the
feature extraction script. Defining them ONCE here guarantees that the
quantity you plot is exactly the quantity you extract features from.
One canonical definition per quantity = one citation per quantity in
the dissertation methodology chapter.

Every function takes either:
  - a raw message of shape (n_samples, 2) with [I, Q] in the last axis, or
  - a complex 1-D array z = I + jQ (convert with to_complex()).
"""

import numpy as np

# Sampling rate of the Watch This Space / SatIQ receivers (25 MS/s).
SAMPLE_RATE_HZ = 25_000_000

def to_complex(message: np.ndarray) -> np.ndarray:
    """
    Convert a raw (n_samples, 2) [I, Q] message into a complex 1-D array.

    z[t] = I[t] + j*Q[t]

    Working with the complex array lets numpy compute amplitude and phase
    directly (np.abs, np.angle) instead of manual sqrt/arctan2 — identical
    results, less code, fewer places for a definition to drift.
    """
    return message[:, 0].astype(np.float64) + 1j * message[:, 1].astype(np.float64)

def amplitude(z: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude |z| = sqrt(I^2 + Q^2)."""
    return np.abs(z)

def phase(z: np.ndarray) -> np.ndarray:
    """Instantaneous phase angle(z) = arctan2(Q, I), in radians (-pi, pi]."""
    return np.angle(z)

def fft_spectrum(z: np.ndarray, fs: float = SAMPLE_RATE_HZ):
    """
    Magnitude spectrum of the complex baseband signal.

    Returns (freqs_hz, magnitude), both fftshift-ed so 0 Hz (the carrier)
    sits in the middle of the array. This is THE spectrum: the plotting
    scripts and the spectral features all use this exact function, so a
    'peak frequency' in features.csv is the peak of the same curve you
    see in the figures.
    """
    mag = np.abs(np.fft.fftshift(np.fft.fft(z)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(z), d=1.0 / fs))
    return freqs, mag

def discover_segments(data_dir, column: str) -> list:
    """
    Return the sorted list of .npy segment files for one dataset column,
    e.g. discover_segments(DATA_DIR, "samples") ->
         [samples_000.npy, samples_001.npy, ...]
    """
    files = sorted(data_dir.glob(f"{column}_*.npy"))
    if not files:
        raise FileNotFoundError(f"No '{column}_*.npy' files found in {data_dir}")
    return files

def segment_offsets(segment_files: list) -> np.ndarray:
    """
    Cumulative row offsets across segment files, read from ACTUAL shapes
    (memory-mapped, so this costs no RAM and no real I/O).

    Example with segments of length [10000, 10000, 9500]:
        returns [0, 10000, 20000, 29500]

    Global message index g lives in segment s where
        offsets[s] <= g < offsets[s+1]
    and its row within that segment is g - offsets[s].

    We never assume segments hold exactly 10,000 rows — partial final
    segments are common in chunked datasets, and hardcoding the segment
    length would silently mis-map every message after the short segment.
    """
    lengths = [np.load(f, mmap_mode="r").shape[0] for f in segment_files]
    return np.concatenate([[0], np.cumsum(lengths)])
