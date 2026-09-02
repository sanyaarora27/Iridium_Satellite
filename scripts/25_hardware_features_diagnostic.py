#!/usr/bin/env python3
"""
25_hardware_features_diagnostic.py
===================================
Tests whether hardware-targeted feature extraction can discriminate satellites
when raw IQ stats and CNN both fail.

Feature groups tested:
  1. CFO (carrier frequency offset) — oscillator-specific
  2. IQ imbalance (gain & phase mismatch) — mixer-specific
  3. Higher-order cumulants (C20, C21, C40, C41, C42) — nonlinear distortion
  4. Instantaneous frequency statistics — phase noise / oscillator drift
  5. Cyclostationary features at Iridium symbol rate — modulator distortion
  6. All combined

Each group is tested within-session (random 80/20) AND cross-session (GroupKFold)
to distinguish "real signal" from "channel artefact".
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_SATS = [51, 85, 87, 92, 109]
MAX_VALID_TS = 5e15
SESSION_BOUNDARIES = [1940e12, 1995e12]
SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("Loading all segments...")
print("=" * 70)

all_iq, all_labels, all_ts = [], [], []
for seg in range(5):
    samples = np.load(DATA_DIR / f"samples_{seg:03d}.npy")
    sats = np.load(DATA_DIR / f"ra_sat_{seg:03d}.npy")
    ts = np.load(DATA_DIR / f"timestamp_{seg:03d}.npy")
    for sat in TARGET_SATS:
        mask = (sats == sat) & (ts < MAX_VALID_TS)
        if mask.sum() == 0:
            continue
        all_iq.append(samples[mask])
        all_labels.append(np.full(mask.sum(), sat))
        all_ts.append(ts[mask])

all_iq = np.concatenate(all_iq)
all_labels = np.concatenate(all_labels)
all_ts = np.concatenate(all_ts)

sat_to_idx = {s: i for i, s in enumerate(TARGET_SATS)}
y = np.array([sat_to_idx[s] for s in all_labels])

# Session groups
sessions = np.array([0 if t < SESSION_BOUNDARIES[0] else
                     1 if t < SESSION_BOUNDARIES[1] else 2 for t in all_ts])

print(f"Total bursts: {len(all_iq)}")
for s in range(3):
    print(f"  Session {chr(65+s)}: {(sessions == s).sum()}")


def extract_cfo_features(iq_batch):
    """
    Carrier Frequency Offset estimation.
    Uses phase-difference method: CFO ∝ mean of arg(x[n] * conj(x[n-1]))
    Also extracts CFO variance and drift (linear trend in phase).
    """
    feats = []
    for burst in iq_batch:
        I, Q = burst[:, 0], burst[:, 1]
        x = I + 1j * Q

        # Phase difference (instantaneous frequency proxy)
        phase_diff = np.angle(x[1:] * np.conj(x[:-1]))

        # CFO estimate (mean phase increment per sample)
        cfo_mean = np.mean(phase_diff)
        cfo_std = np.std(phase_diff)
        cfo_median = np.median(phase_diff)

        # CFO drift: linear regression slope of phase_diff over time
        t = np.arange(len(phase_diff))
        if len(phase_diff) > 1:
            cfo_drift = np.polyfit(t, phase_diff, 1)[0]
        else:
            cfo_drift = 0.0

        # Autocorrelation of phase_diff at lag 1 (oscillator stability)
        pd_centered = phase_diff - cfo_mean
        if np.std(pd_centered) > 1e-10:
            acf1 = np.correlate(pd_centered[:-1], pd_centered[1:])[0] / (
                np.var(pd_centered) * len(pd_centered) + 1e-10)
        else:
            acf1 = 0.0

        feats.append([cfo_mean, cfo_std, cfo_median, cfo_drift, acf1])
    return np.array(feats)

def extract_iq_imbalance_features(iq_batch):
    """
    IQ imbalance: gain mismatch and phase mismatch between I and Q arms.
    These are analog front-end hardware characteristics.
    """
    feats = []
    for burst in iq_batch:
        I, Q = burst[:, 0], burst[:, 1]

        # Gain imbalance: ratio of I power to Q power
        power_I = np.mean(I**2) + 1e-10
        power_Q = np.mean(Q**2) + 1e-10
        gain_imbalance = power_I / power_Q
        gain_imbalance_dB = 10 * np.log10(gain_imbalance)

        # Phase imbalance: deviation from 90° between I and Q
        # Estimated via correlation
        iq_corr = np.corrcoef(I, Q)[0, 1]
        # In ideal case, I and Q are orthogonal (corr=0)
        # Phase imbalance causes nonzero correlation
        phase_imbalance = np.arcsin(np.clip(iq_corr, -1, 1))

        # DC offset (mixer leakage)
        dc_I = np.mean(I)
        dc_Q = np.mean(Q)
        dc_magnitude = np.sqrt(dc_I**2 + dc_Q**2)

        # Second-order stats of imbalance (variability within burst)
        # Split burst into 10 segments and measure imbalance variation
        n_seg = 10
        seg_len = len(I) // n_seg
        gain_vars, phase_vars = [], []
        for i in range(n_seg):
            s, e = i * seg_len, (i + 1) * seg_len
            pI = np.mean(I[s:e]**2) + 1e-10
            pQ = np.mean(Q[s:e]**2) + 1e-10
            gain_vars.append(pI / pQ)
            c = np.corrcoef(I[s:e], Q[s:e])[0, 1]
            phase_vars.append(c if np.isfinite(c) else 0)
        gain_stability = np.std(gain_vars)
        phase_stability = np.std(phase_vars)

        feats.append([gain_imbalance, gain_imbalance_dB, phase_imbalance,
                      dc_I, dc_Q, dc_magnitude, gain_stability, phase_stability])
    return np.array(feats)

def extract_higher_order_cumulants(iq_batch):
    """
    Higher-order cumulants (2nd and 4th order).
    These capture nonlinear distortion from the power amplifier and mixer.
    Theoretically invariant to additive Gaussian noise.

    C20 = E[x^2],  C21 = E[|x|^2]
    C40 = cum4(x, x, x, x),  C41 = cum4(x, x, x, x*),  C42 = cum4(x, x, x*, x*)
    """
    feats = []
    for burst in iq_batch:
        I, Q = burst[:, 0], burst[:, 1]
        x = I + 1j * Q

        # Normalise by power to remove channel gain effect
        power = np.mean(np.abs(x)**2) + 1e-10
        x_norm = x / np.sqrt(power)

        # 2nd-order moments
        C20 = np.mean(x_norm**2)
        C21 = np.mean(np.abs(x_norm)**2)  # should be ~1 after normalisation

        # 4th-order cumulants
        M40 = np.mean(x_norm**4)
        M41 = np.mean((x_norm**3) * np.conj(x_norm))
        M42 = np.mean((np.abs(x_norm)**2) * x_norm**2)
        M22 = np.mean(np.abs(x_norm)**4)

        # Cumulants (subtract Gaussian contribution)
        C40 = M40 - 3 * C20**2
        C41 = M41 - 3 * C20 * C21
        C42 = M22 - np.abs(C20)**2 - 2 * C21**2

        # Use magnitude and phase of complex cumulants
        feats.append([
            np.abs(C20), np.angle(C20),
            np.real(C21),
            np.abs(C40), np.angle(C40),
            np.abs(C41), np.angle(C41),
            np.abs(C42), np.angle(C42) if np.abs(C42) > 1e-10 else 0,
        ])
    return np.array(feats)

def extract_instantaneous_freq_features(iq_batch):
    """
    Instantaneous frequency via analytic signal (Hilbert transform).
    Captures oscillator drift, phase noise, and frequency instability
    at a finer resolution than FFT-based features.
    """
    from scipy.signal import hilbert
    feats = []
    for burst in iq_batch:
        I, Q = burst[:, 0], burst[:, 1]
        x = I + 1j * Q

        # Instantaneous phase (unwrapped)
        inst_phase = np.unwrap(np.angle(x))

        # Instantaneous frequency (derivative of phase)
        inst_freq = np.diff(inst_phase)

        # Statistics of instantaneous frequency
        if_mean = np.mean(inst_freq)
        if_std = np.std(inst_freq)
        if_skew = np.mean((inst_freq - if_mean)**3) / (if_std**3 + 1e-10)
        if_kurt = np.mean((inst_freq - if_mean)**4) / (if_std**4 + 1e-10)
        if_median = np.median(inst_freq)
        if_iqr = np.percentile(inst_freq, 75) - np.percentile(inst_freq, 25)

        # Phase noise: variance of inst_freq in short windows
        win = 100
        n_wins = len(inst_freq) // win
        if n_wins > 0:
            local_vars = [np.var(inst_freq[i*win:(i+1)*win]) for i in range(n_wins)]
            phase_noise_mean = np.mean(local_vars)
            phase_noise_std = np.std(local_vars)
        else:
            phase_noise_mean = np.var(inst_freq)
            phase_noise_std = 0.0

        # Allan-variance-like metric: variance of frequency differences
        freq_diff = np.diff(inst_freq)
        allan_var = 0.5 * np.mean(freq_diff**2)

        # Frequency drift: slope of inst_freq over burst
        t = np.arange(len(inst_freq))
        freq_drift = np.polyfit(t, inst_freq, 1)[0]

        feats.append([if_mean, if_std, if_skew, if_kurt, if_median, if_iqr,
                      phase_noise_mean, phase_noise_std, allan_var, freq_drift])
    return np.array(feats)

def extract_cyclostationary_features(iq_batch, symbol_rate=25000, sample_rate=250000):
    """
    Cyclostationary features at the Iridium symbol rate.
    The cyclic autocorrelation at the symbol rate captures modulator-specific
    distortion that's different from channel effects.
    """
    alpha = symbol_rate / sample_rate  # cyclic frequency
    feats = []
    for burst in iq_batch:
        I, Q = burst[:, 0], burst[:, 1]
        x = I + 1j * Q
        N = len(x)
        t = np.arange(N)

        # Cyclic autocorrelation at alpha for several lags
        cycle_feats = []
        for tau in [0, 1, 2, 4, 8, 16]:
            if tau == 0:
                caf = np.mean(x * np.conj(x) * np.exp(-1j * 2 * np.pi * alpha * t))
            else:
                x_shifted = np.roll(x, tau)
                caf = np.mean(x[tau:] * np.conj(x_shifted[tau:]) *
                              np.exp(-1j * 2 * np.pi * alpha * t[tau:]))
            cycle_feats.extend([np.abs(caf), np.angle(caf)])

        # Also at 2*alpha (second harmonic)
        for tau in [0, 1]:
            caf2 = np.mean(x * np.conj(x) *
                           np.exp(-1j * 2 * np.pi * 2 * alpha * t))
            cycle_feats.extend([np.abs(caf2), np.angle(caf2)])

        # Spectral coherence magnitude at alpha
        # Simplified: ratio of cyclic to non-cyclic power
        noncyclic = np.mean(np.abs(x)**2)
        caf0 = np.abs(np.mean(x * np.conj(x) * np.exp(-1j * 2 * np.pi * alpha * t)))
        coherence = caf0 / (noncyclic + 1e-10)
        cycle_feats.append(coherence)

        feats.append(cycle_feats)
    return np.array(feats)

print("\n" + "=" * 70)
print("Extracting hardware-targeted features...")
print("=" * 70)

print("  CFO features...", end=" ", flush=True)
feat_cfo = extract_cfo_features(all_iq)
print(f"shape {feat_cfo.shape}")

print("  IQ imbalance features...", end=" ", flush=True)
feat_iqimb = extract_iq_imbalance_features(all_iq)
print(f"shape {feat_iqimb.shape}")

print("  Higher-order cumulants...", end=" ", flush=True)
feat_cum = extract_higher_order_cumulants(all_iq)
print(f"shape {feat_cum.shape}")

print("  Instantaneous frequency...", end=" ", flush=True)
feat_instf = extract_instantaneous_freq_features(all_iq)
print(f"shape {feat_instf.shape}")

print("  Cyclostationary features...", end=" ", flush=True)
feat_cyclo = extract_cyclostationary_features(all_iq)
print(f"shape {feat_cyclo.shape}")

# Combined
feat_all = np.hstack([feat_cfo, feat_iqimb, feat_cum, feat_instf, feat_cyclo])
print(f"\n  Combined: {feat_all.shape[1]} features total")

feature_groups = {
    "CFO (5 feat)":              feat_cfo,
    "IQ imbalance (8 feat)":     feat_iqimb,
    "Cumulants (9 feat)":        feat_cum,
    "Inst. frequency (10 feat)": feat_instf,
    "Cyclostationary (17 feat)": feat_cyclo,
    "CFO + IQ imbalance (13)":   np.hstack([feat_cfo, feat_iqimb]),
    "CFO + IQ + Cumulants (22)": np.hstack([feat_cfo, feat_iqimb, feat_cum]),
    "ALL COMBINED (49 feat)":    feat_all,
}

for name in feature_groups:
    feature_groups[name] = np.nan_to_num(feature_groups[name], nan=0.0,
                                          posinf=0.0, neginf=0.0)


def evaluate_within_session(feats, y, session_mask, session_name):
    """80/20 stratified split within a single session."""
    idx = np.where(session_mask)[0]
    if len(idx) < 50:
        return None, None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(sss.split(feats[idx], y[idx]))
    tr_idx, te_idx = idx[tr], idx[te]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feats[tr_idx])
    X_te = scaler.transform(feats[te_idx])

    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y[tr_idx])
    return accuracy_score(y[tr_idx], rf.predict(X_tr)), accuracy_score(y[te_idx], rf.predict(X_te))

def evaluate_cross_session(feats, y, groups):
    """3-fold GroupKFold by session."""
    gkf = GroupKFold(n_splits=3)
    all_preds, all_true = [], []
    for train_idx, test_idx in gkf.split(feats, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feats[train_idx])
        X_te = scaler.transform(feats[test_idx])

        rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                     random_state=SEED, n_jobs=-1)
        rf.fit(X_tr, y[train_idx])
        all_preds.append(rf.predict(X_te))
        all_true.append(y[test_idx])

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    return accuracy_score(all_true, all_preds), f1_score(all_true, all_preds, average="macro")

print("\n" + "=" * 70)
print("WITHIN-SESSION evaluation (Session A, 80/20 split)")
print("Can the features discriminate at all?")
print("=" * 70)
print(f"{'Feature group':<35} {'Train acc':>10} {'Test acc':>10}")
print("-" * 57)

for name, feats in feature_groups.items():
    tr, te = evaluate_within_session(feats, y, sessions == 0, "A")
    marker = " ***" if te is not None and te > 0.25 else ""
    print(f"{name:<35} {tr:>10.3f} {te:>10.3f}{marker}")

print("\n" + "=" * 70)
print("CROSS-SESSION evaluation (3-fold GroupKFold)")
print("Does the signal generalise across recording sessions?")
print("=" * 70)
print(f"{'Feature group':<35} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 57)

cross_results = {}
for name, feats in feature_groups.items():
    acc, f1 = evaluate_cross_session(feats, y, sessions)
    marker = " ***" if acc > 0.25 else ""
    print(f"{name:<35} {acc:>10.3f} {f1:>10.3f}{marker}")
    cross_results[name] = (acc, f1)

print("\n" + "=" * 70)
print("GRADIENT BOOSTING on combined features (cross-session)")
print("=" * 70)

scaler = StandardScaler()
gkf = GroupKFold(n_splits=3)
all_preds, all_true = [], []
for train_idx, test_idx in gkf.split(feat_all, y, groups=sessions):
    X_tr = scaler.fit_transform(feat_all[train_idx])
    X_te = scaler.transform(feat_all[test_idx])
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                     learning_rate=0.05, random_state=SEED)
    gb.fit(X_tr, y[train_idx])
    all_preds.append(gb.predict(X_te))
    all_true.append(y[test_idx])

all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)
gb_acc = accuracy_score(all_true, all_preds)
gb_f1 = f1_score(all_true, all_preds, average="macro")
print(f"  GBM cross-session: accuracy={gb_acc:.4f}, macro-F1={gb_f1:.4f}")

print("\n" + "=" * 70)
print("TOP 15 FEATURES by importance (Random Forest, full dataset)")
print("=" * 70)

feat_names = (
    ["cfo_mean", "cfo_std", "cfo_median", "cfo_drift", "cfo_acf1"] +
    ["gain_imb", "gain_imb_dB", "phase_imb", "dc_I", "dc_Q", "dc_mag",
     "gain_stability", "phase_stability"] +
    ["abs_C20", "angle_C20", "C21", "abs_C40", "angle_C40",
     "abs_C41", "angle_C41", "abs_C42", "angle_C42"] +
    ["if_mean", "if_std", "if_skew", "if_kurt", "if_median", "if_iqr",
     "phasenoise_mean", "phasenoise_std", "allan_var", "freq_drift"] +
    [f"cyc_tau{t}_{x}" for t in [0,1,2,4,8,16] for x in ["abs","angle"]] +
    [f"cyc2_tau{t}_{x}" for t in [0,1] for x in ["abs","angle"]] +
    ["spectral_coherence"]
)

# Fit on all data for importance ranking
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_all)
rf_full = RandomForestClassifier(n_estimators=500, max_depth=15,
                                  random_state=SEED, n_jobs=-1)
rf_full.fit(X_scaled, y)
importances = rf_full.feature_importances_

# Pad or trim names if needed
if len(feat_names) < feat_all.shape[1]:
    feat_names.extend([f"feat_{i}" for i in range(len(feat_names), feat_all.shape[1])])

top_idx = np.argsort(importances)[::-1][:15]
for rank, i in enumerate(top_idx):
    print(f"  {rank+1:2d}. {feat_names[i]:<25s} importance={importances[i]:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
best_name = max(cross_results, key=lambda k: cross_results[k][0])
best_acc, best_f1 = cross_results[best_name]

print(f"\n  Chance level: 20.0%")
print(f"  Best cross-session result: {best_name}")
print(f"    Accuracy: {best_acc:.1%}")
print(f"    Macro F1: {best_f1:.4f}")

if best_acc > 0.30:
    print(f"\n  ✓ ABOVE CHANCE — hardware features carry discriminative signal!")
    print(f"    → Feed these features into CNN/MLP for the thesis model.")
elif best_acc > 0.25:
    print(f"\n  ~ MARGINAL — slight signal detected but weak.")
    print(f"    → May improve with feature selection or more data.")
else:
    print(f"\n  ✗ AT CHANCE — no hardware-discriminative signal found.")
    print(f"    → Strong negative result. Document as thesis finding.")
    print(f"    → Proceed to higher-layer integration framework.")
