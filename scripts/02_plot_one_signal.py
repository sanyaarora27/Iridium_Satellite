from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE_HZ = 25_000_000
MESSAGE_INDEX  = 0

samples_file = DATA_DIR / "samples_000.npy"
print(f"Loading {samples_file.name}...")

samples = np.load(samples_file)
print(f"  Loaded array of shape {samples.shape}, dtype {samples.dtype}")

one_message = samples[MESSAGE_INDEX]
print(f"  Picked message #{MESSAGE_INDEX}, shape {one_message.shape}")

i_values = one_message[:, 0]
q_values = one_message[:, 1]

n_samples = len(i_values)
time_seconds = np.arange(n_samples) / SAMPLE_RATE_HZ
time_microseconds = time_seconds * 1e6

amplitude = np.sqrt(i_values**2 + q_values**2)

phase = np.arctan2(q_values, i_values)

complex_signal = i_values + 1j * q_values
fft_complex    = np.fft.fftshift(np.fft.fft(complex_signal))
fft_magnitude  = np.abs(fft_complex)
fft_freqs_hz   = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1.0 / SAMPLE_RATE_HZ))
fft_freqs_mhz  = fft_freqs_hz / 1e6

fig, axes = plt.subplots(5, 1, figsize=(11, 13))

axes[0].plot(time_microseconds, i_values, color="steelblue", linewidth=0.6)
axes[0].set_ylabel("I  (in-phase)")
axes[0].set_title(f"Real Iridium message - index {MESSAGE_INDEX} from samples_000.npy")
axes[0].grid(alpha=0.3)

axes[1].plot(time_microseconds, q_values, color="darkorange", linewidth=0.6)
axes[1].set_ylabel("Q  (quadrature)")
axes[1].grid(alpha=0.3)

axes[2].plot(time_microseconds, amplitude, color="seagreen", linewidth=0.6)
axes[2].set_ylabel("Amplitude")
axes[2].grid(alpha=0.3)

axes[3].plot(time_microseconds, phase, color="purple", linewidth=0.6)
axes[3].set_ylabel("Phase  (rad)")
axes[3].set_xlabel("Time  (microseconds)")
axes[3].grid(alpha=0.3)

axes[4].plot(fft_freqs_mhz, fft_magnitude, color="crimson", linewidth=0.6)
axes[4].set_ylabel("FFT magnitude")
axes[4].set_xlabel("Frequency offset from carrier  (MHz)")
axes[4].set_title("Frequency-domain spectrum (FFT of the complex signal)")
axes[4].grid(alpha=0.3)

plt.tight_layout()

output_path = OUTPUT_DIR / "one_iq_message.png"
plt.savefig(output_path, dpi=120, bbox_inches="tight")
print(f"  Saved figure: {output_path}")

plt.show()
