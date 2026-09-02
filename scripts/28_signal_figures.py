#!/usr/bin/env python3
"""
28_signal_figures.py — Signal visualisation for thesis
======================================================
Generates two sets of figures:
  1. Single satellite detailed view (4 panels: I/Q, amplitude, phase, spectrum)
  2. All 5 satellites compared (overlay / side-by-side)

These go in the dataset/methodology chapter to show what the IQ data
looks like and visually demonstrate why satellites are hard to distinguish.

Outputs:
  outputs/figures/signal_single_sat51.png
  outputs/figures/signal_all5_comparison.png
  outputs/figures/signal_all5_spectrum.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"

OUT_FIGS.mkdir(parents=True, exist_ok=True)

TARGET_SATS = [51, 85, 87, 92, 109]
SAT_COLORS = {51: "#e74c3c", 85: "#3498db", 87: "#2ecc71", 92: "#9b59b6", 109: "#f39c12"}
MAX_TS = 5e15

print("Loading sample bursts...")
samples = np.load(DATA_DIR / "samples_000.npy")
sats = np.load(DATA_DIR / "ra_sat_000.npy")
ts = np.load(DATA_DIR / "timestamp_000.npy")

bursts = {}
for sat in TARGET_SATS:
    mask = (sats == sat) & (ts < MAX_TS)
    idx = np.where(mask)[0]
    # Pick a burst from the middle (more representative than first/last)
    bursts[sat] = samples[idx[len(idx) // 2]]
    print(f"  Sat {sat}: selected burst {idx[len(idx)//2]}, shape {bursts[sat].shape}")

# Sampling rate (Iridium downlink ≈ 25 ksps, SatIQ typically oversampled)
# From dataset: 11000 samples in 440 µs → 25 Msps
SAMPLE_RATE = 25e6  # Hz (adjust if different)
T_US = np.arange(11000) / SAMPLE_RATE * 1e6  # time in microseconds
FREQ_KHZ = np.fft.fftfreq(11000, d=1/SAMPLE_RATE) / 1e3  # freq in kHz

# Show only first 2000 samples in time plots for clarity
SHOW_N = 2000
t_show = T_US[:SHOW_N]

# FIGURE 1: Single satellite detailed view (Sat 51)
print("\nGenerating single-satellite figure (Sat 51)...")

sat = 51
burst = bursts[sat]
I, Q = burst[:, 0], burst[:, 1]
amp = np.sqrt(I**2 + Q**2)
phase = np.unwrap(np.arctan2(Q, I))
fft_mag = np.abs(np.fft.fft(I + 1j * Q))
fft_mag_dB = 20 * np.log10(fft_mag / (np.max(fft_mag) + 1e-10) + 1e-10)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: I and Q waveforms
ax = axes[0, 0]
ax.plot(t_show, I[:SHOW_N], alpha=0.7, linewidth=0.5, color="#e74c3c", label="I (in-phase)")
ax.plot(t_show, Q[:SHOW_N], alpha=0.7, linewidth=0.5, color="#3498db", label="Q (quadrature)")
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Amplitude")
ax.set_title("(a) Raw I/Q Waveform")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Panel 2: Amplitude envelope
ax = axes[0, 1]
ax.plot(t_show, amp[:SHOW_N], linewidth=0.5, color="#2ecc71")
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Amplitude")
ax.set_title("(b) Amplitude Envelope |I + jQ|")
ax.grid(True, alpha=0.2)

# Panel 3: Phase trajectory
ax = axes[1, 0]
ax.plot(t_show, phase[:SHOW_N], linewidth=0.5, color="#9b59b6")
ax.set_xlabel("Time (µs)")
ax.set_ylabel("Phase (radians)")
ax.set_title("(c) Unwrapped Phase")
ax.grid(True, alpha=0.2)

# Panel 4: Frequency spectrum
ax = axes[1, 1]
ax.plot(FREQ_KHZ, fft_mag_dB, linewidth=0.5, color="#f39c12")
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("(d) Frequency Spectrum")
ax.set_ylim(bottom=-80)
ax.grid(True, alpha=0.2)

fig.suptitle(f"Satellite {sat} — Single Burst Signal Characteristics", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FIGS / "signal_single_sat51.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'signal_single_sat51.png'}")

# FIGURE 2: All 5 satellites compared — time domain
print("\nGenerating 5-satellite comparison figure...")

fig, axes = plt.subplots(5, 4, figsize=(18, 18))

for row, sat in enumerate(TARGET_SATS):
    burst = bursts[sat]
    I, Q = burst[:, 0], burst[:, 1]
    amp = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.arctan2(Q, I))
    fft_mag = np.abs(np.fft.fft(I + 1j * Q))
    fft_mag_dB = 20 * np.log10(fft_mag / (np.max(fft_mag) + 1e-10) + 1e-10)
    color = SAT_COLORS[sat]

    # I/Q
    axes[row, 0].plot(t_show, I[:SHOW_N], alpha=0.7, linewidth=0.4, color=color)
    axes[row, 0].plot(t_show, Q[:SHOW_N], alpha=0.5, linewidth=0.4, color=color, linestyle="--")
    axes[row, 0].set_ylabel(f"Sat {sat}", fontsize=11, fontweight="bold")
    if row == 0:
        axes[row, 0].set_title("I/Q Waveform")
    if row == 4:
        axes[row, 0].set_xlabel("Time (µs)")

    # Amplitude
    axes[row, 1].plot(t_show, amp[:SHOW_N], linewidth=0.4, color=color)
    if row == 0:
        axes[row, 1].set_title("Amplitude")
    if row == 4:
        axes[row, 1].set_xlabel("Time (µs)")

    # Phase
    axes[row, 2].plot(t_show, phase[:SHOW_N], linewidth=0.4, color=color)
    if row == 0:
        axes[row, 2].set_title("Phase")
    if row == 4:
        axes[row, 2].set_xlabel("Time (µs)")

    # Spectrum
    axes[row, 3].plot(FREQ_KHZ, fft_mag_dB, linewidth=0.4, color=color)
    axes[row, 3].set_ylim(-80, 5)
    if row == 0:
        axes[row, 3].set_title("Spectrum (dB)")
    if row == 4:
        axes[row, 3].set_xlabel("Frequency (kHz)")

    for col in range(4):
        axes[row, col].grid(True, alpha=0.15)
        axes[row, col].tick_params(labelsize=8)

fig.suptitle("All 5 Satellites — Signal Comparison\n"
             "Visual similarity confirms lack of transmitter-specific features",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FIGS / "signal_all5_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'signal_all5_comparison.png'}")

# FIGURE 3: All 5 spectra overlaid (most impactful comparison)
print("\nGenerating overlaid spectrum comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overlaid I waveforms (zoomed to 500 samples)
ax = axes[0]
ZOOM = 500
t_zoom = T_US[:ZOOM]
for sat in TARGET_SATS:
    I = bursts[sat][:ZOOM, 0]
    ax.plot(t_zoom, I, linewidth=0.6, alpha=0.7, color=SAT_COLORS[sat], label=f"Sat {sat}")
ax.set_xlabel("Time (µs)")
ax.set_ylabel("I amplitude")
ax.set_title("(a) In-Phase Waveforms (first 500 samples)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Overlaid amplitude histograms
ax = axes[1]
for sat in TARGET_SATS:
    amp = np.sqrt(bursts[sat][:, 0]**2 + bursts[sat][:, 1]**2)
    ax.hist(amp, bins=50, alpha=0.4, color=SAT_COLORS[sat], label=f"Sat {sat}",
            density=True, histtype="stepfilled")
ax.set_xlabel("Amplitude")
ax.set_ylabel("Density")
ax.set_title("(b) Amplitude Distributions")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Overlaid spectra
ax = axes[2]
for sat in TARGET_SATS:
    I, Q = bursts[sat][:, 0], bursts[sat][:, 1]
    fft_mag = np.abs(np.fft.fft(I + 1j * Q))
    fft_dB = 20 * np.log10(fft_mag / (np.max(fft_mag) + 1e-10) + 1e-10)
    ax.plot(FREQ_KHZ, fft_dB, linewidth=0.7, alpha=0.7, color=SAT_COLORS[sat],
            label=f"Sat {sat}")
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("(c) Frequency Spectra Overlaid")
ax.set_ylim(-80, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

fig.suptitle("Satellite Signal Overlap — Why Classification Fails\n"
             "All five satellites produce nearly identical waveforms, amplitudes, and spectra",
             fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig(OUT_FIGS / "signal_all5_spectrum.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'signal_all5_spectrum.png'}")

print("\nGenerating individual satellite figures...")
for sat in TARGET_SATS:
    burst = bursts[sat]
    I, Q = burst[:, 0], burst[:, 1]
    amp = np.sqrt(I**2 + Q**2)
    phase = np.unwrap(np.arctan2(Q, I))
    fft_mag = np.abs(np.fft.fft(I + 1j * Q))
    fft_dB = 20 * np.log10(fft_mag / (np.max(fft_mag) + 1e-10) + 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t_show, I[:SHOW_N], lw=0.5, color=SAT_COLORS[sat])
    axes[0, 0].plot(t_show, Q[:SHOW_N], lw=0.5, color=SAT_COLORS[sat], alpha=0.5, ls="--")
    axes[0, 0].set_title("I/Q Waveform"); axes[0, 0].set_xlabel("Time (µs)")

    axes[0, 1].plot(t_show, amp[:SHOW_N], lw=0.5, color=SAT_COLORS[sat])
    axes[0, 1].set_title("Amplitude"); axes[0, 1].set_xlabel("Time (µs)")

    axes[1, 0].plot(t_show, phase[:SHOW_N], lw=0.5, color=SAT_COLORS[sat])
    axes[1, 0].set_title("Phase"); axes[1, 0].set_xlabel("Time (µs)")

    axes[1, 1].plot(FREQ_KHZ, fft_dB, lw=0.5, color=SAT_COLORS[sat])
    axes[1, 1].set_title("Spectrum"); axes[1, 1].set_xlabel("Freq (kHz)"); axes[1, 1].set_ylim(-80, 5)

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"Satellite {sat} — Signal Characteristics", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / f"signal_single_sat{sat}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"  Saved: individual figures for all 5 satellites")
print("\nDone.")
