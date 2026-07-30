from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- PATHS ----------------------------------------------------------------
# This script lives in scripts/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---------------------------------------------------------------
# The Iridium signal was sampled at 25 million samples per second.
# This is the value from the original Watch This Space / SatIQ paper.
SAMPLE_RATE_HZ = 25_000_000

# Which message in the file to plot. 0 = the first one.
MESSAGE_INDEX  = 0


# --- STEP 1: LOAD ONE FILE ------------------------------------------------
# samples_000.npy is the first segment of IQ samples in the dataset.
# Each segment contains around 10,000 messages.
samples_file = DATA_DIR / "samples_000.npy"
print(f"Loading {samples_file.name}...")

samples = np.load(samples_file)
print(f"  Loaded array of shape {samples.shape}, dtype {samples.dtype}")
# Expected shape: (n_messages, signal_length, 2)
# where the last dimension is [I, Q]


# --- STEP 2: PICK ONE MESSAGE ---------------------------------------------
one_message = samples[MESSAGE_INDEX]
print(f"  Picked message #{MESSAGE_INDEX}, shape {one_message.shape}")

# Split into I and Q components
i_values = one_message[:, 0]  # Channel 0 = I (real part)
q_values = one_message[:, 1]  # Channel 1 = Q (imaginary part)

# Time axis in seconds: each sample is 1/SAMPLE_RATE_HZ seconds apart
n_samples = len(i_values)
time_seconds = np.arange(n_samples) / SAMPLE_RATE_HZ
time_microseconds = time_seconds * 1e6  # Convert to microseconds for readability


# --- STEP 3: COMPUTE DERIVED QUANTITIES -----------------------------------
# Amplitude: how strong is the signal at each moment in time
amplitude = np.sqrt(i_values**2 + q_values**2)

# Phase: angle of the (I, Q) vector at each moment
# np.arctan2(Q, I) gives the angle in radians from -pi to +pi
phase = np.arctan2(q_values, i_values)

# FFT: convert from time domain to frequency domain
# np.fft.fft produces complex numbers; we take the magnitude (absolute value)
# np.fft.fftshift centres the spectrum so that 0 Hz is in the middle
complex_signal = i_values + 1j * q_values  # combine I and Q into complex array
fft_complex    = np.fft.fftshift(np.fft.fft(complex_signal))
fft_magnitude  = np.abs(fft_complex)
# Frequencies corresponding to each FFT bin (also shifted)
fft_freqs_hz   = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1.0 / SAMPLE_RATE_HZ))
fft_freqs_mhz  = fft_freqs_hz / 1e6  # Convert to MHz


# --- STEP 4: PLOT EVERYTHING ----------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(11, 13))

# Plot 1: I over time
axes[0].plot(time_microseconds, i_values, color="steelblue", linewidth=0.6)
axes[0].set_ylabel("I  (in-phase)")
axes[0].set_title(f"Real Iridium message - index {MESSAGE_INDEX} from samples_000.npy")
axes[0].grid(alpha=0.3)

# Plot 2: Q over time
axes[1].plot(time_microseconds, q_values, color="darkorange", linewidth=0.6)
axes[1].set_ylabel("Q  (quadrature)")
axes[1].grid(alpha=0.3)

# Plot 3: Amplitude over time
axes[2].plot(time_microseconds, amplitude, color="seagreen", linewidth=0.6)
axes[2].set_ylabel("Amplitude")
axes[2].grid(alpha=0.3)

# Plot 4: Phase over time
axes[3].plot(time_microseconds, phase, color="purple", linewidth=0.6)
axes[3].set_ylabel("Phase  (rad)")
axes[3].set_xlabel("Time  (microseconds)")
axes[3].grid(alpha=0.3)

# Plot 5: FFT magnitude spectrum
axes[4].plot(fft_freqs_mhz, fft_magnitude, color="crimson", linewidth=0.6)
axes[4].set_ylabel("FFT magnitude")
axes[4].set_xlabel("Frequency offset from carrier  (MHz)")
axes[4].set_title("Frequency-domain spectrum (FFT of the complex signal)")
axes[4].grid(alpha=0.3)

plt.tight_layout()

# Save the figure to disk
output_path = OUTPUT_DIR / "one_iq_message.png"
plt.savefig(output_path, dpi=120, bbox_inches="tight")
print(f"  Saved figure: {output_path}")

# Show the figure on screen
plt.show()
