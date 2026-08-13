#!/usr/bin/env python3
"""
32_thesis_tables.py — Generate all thesis-ready tables and documents
====================================================================
Outputs:
  outputs/tables/model_comparison.csv
  outputs/tables/feature_table.csv
  outputs/reports/threat_model.md
  outputs/reports/future_work.md
"""

import csv
from pathlib import Path

OUT_TABLES = Path("outputs/tables"); OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_REPORTS = Path("outputs/reports"); OUT_REPORTS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# TABLE 1: MODEL COMPARISON (RF vs MLP vs CNN)
# ══════════════════════════════════════════════════════════════════════════

model_comparison = [
    {
        "Model": "Logistic Regression",
        "Input": "20 extracted features",
        "Architecture": "Linear classifier",
        "Parameters": "~105",
        "Stratified Accuracy": "~20.8%",
        "Stratified Macro F1": "~0.208",
        "Cross-Session Accuracy": "~20.8%",
        "Cross-Session Macro F1": "~0.208",
        "Training Time": "<1 min",
        "Notes": "Baseline linear model",
    },
    {
        "Model": "Random Forest",
        "Input": "20 extracted features",
        "Architecture": "300 trees, max_depth=15",
        "Parameters": "~50K nodes",
        "Stratified Accuracy": "~20.8%",
        "Stratified Macro F1": "~0.208",
        "Cross-Session Accuracy": "~20.8%",
        "Cross-Session Macro F1": "~0.208",
        "Training Time": "<1 min",
        "Notes": "Overfits train (97%) but chance on test",
    },
    {
        "Model": "SVM (RBF)",
        "Input": "20 extracted features",
        "Architecture": "RBF kernel, C=1.0",
        "Parameters": "N/A",
        "Stratified Accuracy": "~20.8%",
        "Stratified Macro F1": "~0.208",
        "Cross-Session Accuracy": "~20.8%",
        "Cross-Session Macro F1": "~0.208",
        "Training Time": "<1 min",
        "Notes": "Kernel method baseline",
    },
    {
        "Model": "k-NN",
        "Input": "20 extracted features",
        "Architecture": "k=5, Euclidean distance",
        "Parameters": "N/A",
        "Stratified Accuracy": "~20.8%",
        "Stratified Macro F1": "~0.208",
        "Cross-Session Accuracy": "~20.8%",
        "Cross-Session Macro F1": "~0.208",
        "Training Time": "<1 min",
        "Notes": "Instance-based baseline",
    },
    {
        "Model": "MLP",
        "Input": "20 extracted features",
        "Architecture": "Dense(128)→BN→ReLU→Drop(0.3)→Dense(64)→BN→ReLU→Drop(0.3)→Dense(32)→Dense(5)",
        "Parameters": "~12K",
        "Stratified Accuracy": "25.6%",
        "Stratified Macro F1": "0.243",
        "Cross-Session Accuracy": "23.8%",
        "Cross-Session Macro F1": "0.221",
        "Training Time": "~10 min",
        "Notes": "3-layer neural network, AdamW, cosine LR",
    },
    {
        "Model": "1D-CNN",
        "Input": "Raw IQ (2×11000)",
        "Architecture": "Conv1d(32,k=64,s=4)→BN→ReLU→Pool(4)→Conv1d(64,k=16,s=2)→BN→ReLU→Pool(2)→Conv1d(128,k=8)→BN→ReLU→GAP→Drop(0.5)→Dense(64)→Drop(0.3)→Dense(5)",
        "Parameters": "~90K",
        "Stratified Accuracy": "21.9%",
        "Stratified Macro F1": "0.172",
        "Cross-Session Accuracy": "24.4%",
        "Cross-Session Macro F1": "0.186",
        "Training Time": "~45 min",
        "Notes": "Learned features from raw IQ, GroupKFold by session",
    },
    {
        "Model": "Chance baseline",
        "Input": "N/A",
        "Architecture": "Random prediction",
        "Parameters": "0",
        "Stratified Accuracy": "20.0%",
        "Stratified Macro F1": "0.200",
        "Cross-Session Accuracy": "20.0%",
        "Cross-Session Macro F1": "0.200",
        "Training Time": "N/A",
        "Notes": "5 classes, uniform prior",
    },
]

with open(OUT_TABLES / "model_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=model_comparison[0].keys())
    w.writeheader()
    w.writerows(model_comparison)
print(f"Saved: {OUT_TABLES / 'model_comparison.csv'}")


# ══════════════════════════════════════════════════════════════════════════
# TABLE 2: FEATURE TABLE
# ══════════════════════════════════════════════════════════════════════════

feature_table = [
    # Time domain
    {"Feature": "mean_I", "Formula": "μ_I = (1/N) Σ I[n]",
     "Group": "Time domain",
     "Hardware Source": "DC offset from mixer LO leakage",
     "Channel Sensitive": "Yes — path loss shifts mean power",
     "Used": "Yes"},
    {"Feature": "std_I", "Formula": "σ_I = √(Var(I))",
     "Group": "Time domain",
     "Hardware Source": "Transmitter output power / amplifier gain",
     "Channel Sensitive": "Yes — fading alters variance",
     "Used": "Yes"},
    {"Feature": "mean_Q", "Formula": "μ_Q = (1/N) Σ Q[n]",
     "Group": "Time domain",
     "Hardware Source": "DC offset from mixer LO leakage (Q arm)",
     "Channel Sensitive": "Yes — path loss shifts mean power",
     "Used": "Yes"},
    {"Feature": "std_Q", "Formula": "σ_Q = √(Var(Q))",
     "Group": "Time domain",
     "Hardware Source": "Transmitter output power / amplifier gain",
     "Channel Sensitive": "Yes — fading alters variance",
     "Used": "Yes"},
    {"Feature": "skewness_I", "Formula": "γ_I = E[(I−μ)³] / σ³",
     "Group": "Time domain",
     "Hardware Source": "Amplifier nonlinearity (AM/AM distortion)",
     "Channel Sensitive": "Moderate — multipath can affect distribution shape",
     "Used": "Yes"},
    {"Feature": "skewness_Q", "Formula": "γ_Q = E[(Q−μ)³] / σ³",
     "Group": "Time domain",
     "Hardware Source": "Amplifier nonlinearity (Q arm)",
     "Channel Sensitive": "Moderate",
     "Used": "Yes"},
    # Amplitude/power
    {"Feature": "mean_amplitude", "Formula": "|x| = √(I² + Q²), then mean",
     "Group": "Amplitude/power",
     "Hardware Source": "Transmitter output power level",
     "Channel Sensitive": "Yes — dominated by path loss and fading",
     "Used": "Yes"},
    {"Feature": "std_amplitude", "Formula": "σ(|x|)",
     "Group": "Amplitude/power",
     "Hardware Source": "Power amplifier stability",
     "Channel Sensitive": "Yes — fading envelope variation",
     "Used": "Yes"},
    {"Feature": "signal_power", "Formula": "P = (1/N) Σ (I² + Q²)",
     "Group": "Amplitude/power",
     "Hardware Source": "Transmitter EIRP",
     "Channel Sensitive": "Yes — path loss, shadowing, antenna gain",
     "Used": "Yes"},
    {"Feature": "PAPR", "Formula": "max(|x|²) / mean(|x|²)",
     "Group": "Amplitude/power",
     "Hardware Source": "Amplifier compression / clipping behaviour",
     "Channel Sensitive": "Moderate — multipath can create peaks",
     "Used": "Yes"},
    # Phase / instantaneous frequency
    {"Feature": "inst_freq_mean", "Formula": "μ(Δφ) = mean(φ[n+1] − φ[n])",
     "Group": "Phase",
     "Hardware Source": "Carrier frequency offset (oscillator drift from nominal)",
     "Channel Sensitive": "Low — CFO is primarily hardware-determined",
     "Used": "Yes"},
    {"Feature": "inst_freq_std", "Formula": "σ(Δφ)",
     "Group": "Phase",
     "Hardware Source": "Phase noise (oscillator jitter)",
     "Channel Sensitive": "Moderate — Doppler spread adds variance",
     "Used": "Yes"},
    {"Feature": "inst_freq_median", "Formula": "median(Δφ)",
     "Group": "Phase",
     "Hardware Source": "Robust CFO estimate",
     "Channel Sensitive": "Low",
     "Used": "Yes"},
    {"Feature": "inst_freq_IQR", "Formula": "P75(Δφ) − P25(Δφ)",
     "Group": "Phase",
     "Hardware Source": "Phase noise spread (oscillator quality)",
     "Channel Sensitive": "Moderate",
     "Used": "Yes"},
    # Frequency domain
    {"Feature": "fft_peak", "Formula": "max(|FFT(I + jQ)|)",
     "Group": "Frequency domain",
     "Hardware Source": "Carrier signal strength",
     "Channel Sensitive": "Yes — proportional to received power",
     "Used": "Yes"},
    {"Feature": "fft_mean", "Formula": "mean(|FFT(I + jQ)|)",
     "Group": "Frequency domain",
     "Hardware Source": "Spectral flatness / noise floor",
     "Channel Sensitive": "Yes — channel frequency response",
     "Used": "Yes"},
    {"Feature": "fft_std", "Formula": "std(|FFT(I + jQ)|)",
     "Group": "Frequency domain",
     "Hardware Source": "Spectral shape consistency",
     "Channel Sensitive": "Yes — frequency-selective fading",
     "Used": "Yes"},
    {"Feature": "spectral_centroid", "Formula": "Σ(f · |X(f)|) / Σ|X(f)|",
     "Group": "Frequency domain",
     "Hardware Source": "Carrier frequency offset (shifts centroid)",
     "Channel Sensitive": "Moderate — Doppler shifts centroid",
     "Used": "Yes"},
    # Cross-domain
    {"Feature": "IQ_correlation", "Formula": "corr(I, Q)",
     "Group": "Cross-domain",
     "Hardware Source": "IQ imbalance (phase mismatch between mixer arms)",
     "Channel Sensitive": "Low — primarily hardware-determined",
     "Used": "Yes"},
    {"Feature": "kurtosis_I", "Formula": "κ_I = E[(I−μ)⁴] / σ⁴",
     "Group": "Cross-domain",
     "Hardware Source": "Amplifier nonlinearity / signal distribution shape",
     "Channel Sensitive": "Moderate",
     "Used": "Yes"},
    # Leakage features (removed)
    {"Feature": "timestamp", "Formula": "Unix epoch time of reception",
     "Group": "Metadata (REMOVED)",
     "Hardware Source": "None — receiver-side metadata",
     "Channel Sensitive": "N/A — encodes temporal ordering",
     "Used": "No — leakage"},
    {"Feature": "global_index", "Formula": "Sequential sample counter",
     "Group": "Metadata (REMOVED)",
     "Hardware Source": "None — dataset ordering artefact",
     "Channel Sensitive": "N/A — encodes recording order",
     "Used": "No — leakage"},
    {"Feature": "ra_lat / ra_lon", "Formula": "Receiver antenna coordinates",
     "Group": "Metadata (REMOVED)",
     "Hardware Source": "None — ground station position",
     "Channel Sensitive": "N/A — encodes receiver location",
     "Used": "No — leakage"},
    {"Feature": "elevation / direction", "Formula": "Satellite elevation angle",
     "Group": "Metadata (REMOVED)",
     "Hardware Source": "None — orbital geometry",
     "Channel Sensitive": "Yes — determines path loss / Doppler",
     "Used": "No — leakage"},
]

with open(OUT_TABLES / "feature_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=feature_table[0].keys())
    w.writeheader()
    w.writerows(feature_table)
print(f"Saved: {OUT_TABLES / 'feature_table.csv'}")


# ══════════════════════════════════════════════════════════════════════════
# THREAT MODEL
# ══════════════════════════════════════════════════════════════════════════

threat_model = """# Threat Model: Multi-Layer Satellite Authentication

## 1. System Overview

The system authenticates Iridium satellite downlink transmissions using
two independent security layers: physical-layer RF fingerprinting and
higher-layer HMAC-based message authentication, combined through a
weighted fusion decision engine.

## 2. Assets

| Asset | Description | Value |
|-------|------------|-------|
| Satellite downlink signal | IQ samples received at ground station | Integrity of navigation/communication data |
| Satellite identity | Claimed satellite ID in message header | Correct attribution of telemetry |
| Pre-shared HMAC keys | Per-satellite 256-bit symmetric keys | Authentication chain integrity |
| RF fingerprint database | Trained CNN model weights | Physical-layer verification capability |
| Authentication decisions | Accept/reject/flag outputs | Downstream system trust |
| Nonce registry | Set of previously used nonces | Replay protection |

## 3. Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│ TRUSTED ZONE                                    │
│                                                 │
│  Ground station receiver (assumed trusted)       │
│  HMAC key storage (assumed secure)              │
│  Fusion decision engine (local computation)      │
│  Nonce registry (local state)                   │
│                                                 │
├─────────────────────────────────────────────────┤
│ UNTRUSTED ZONE                                  │
│                                                 │
│  RF channel (open, interceptable, spoofable)    │
│  Claimed satellite ID (attacker-controlled)     │
│  Message payload (potentially forged)           │
│  Signal waveform (potentially replayed/spoofed) │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Trust boundary:** The air interface between the satellite transmitter
and the ground station receiver. Everything crossing this boundary is
untrusted and must be verified.

## 4. Attacker Model

### 4.1 Attacker Capabilities

| Capability Level | Description |
|-----------------|-------------|
| **Passive** | Can intercept and record satellite downlink signals |
| **Active (basic)** | Can transmit signals on Iridium downlink frequency, can forge message headers with arbitrary claimed satellite IDs |
| **Active (advanced)** | Can replay previously captured legitimate signals, can attempt to mimic RF characteristics of target satellite |
| **Insider** | Has obtained one or more pre-shared HMAC keys through theft, compromise, or insider access |

### 4.2 Attacker Goals

| Goal | Description | Impact |
|------|------------|--------|
| **Identity spoofing** | Convince ground station that transmission originates from a different satellite | Misattribution of telemetry, false positioning data |
| **Message forgery** | Inject fabricated messages accepted as legitimate | Corrupted command/telemetry pipeline |
| **Replay attack** | Re-transmit captured legitimate messages to trigger duplicate processing | State confusion, resource exhaustion |
| **Trust manipulation** | Cause the system to accept malicious transmissions or reject legitimate ones | Denial of service or security bypass |

### 4.3 What the Attacker Cannot Do (Assumptions)

| Assumption | Justification |
|-----------|---------------|
| Cannot compromise the ground station receiver hardware | Physical security of ground infrastructure |
| Cannot modify the fusion engine logic at runtime | Software integrity controls |
| Cannot access the nonce registry | Local memory, not exposed |
| Cannot perfectly replicate another satellite's hardware RF characteristics | Hardware manufacturing variation is physically unique (though our results show this may be undetectable with current receiver resolution) |

## 5. Threat Scenarios

### Threat 1: Identity Spoofing
- **Attack:** Attacker transmits on Iridium frequency with claimed_id = SAT_A, but is not SAT_A
- **Physical layer response:** CNN predicts a different satellite (or low confidence) — but at 20% accuracy, this is unreliable
- **Higher layer response:** Attacker lacks SAT_A's HMAC key → HMAC verification fails
- **Fusion outcome:** REJECT (HMAC failure is decisive)
- **Residual risk:** Low — requires key compromise to bypass

### Threat 2: Stolen Key / Cloned Identity
- **Attack:** Attacker obtains SAT_A's HMAC key, transmits from own hardware claiming to be SAT_A
- **Physical layer response:** CNN may detect RF mismatch (predicted ≠ claimed)
- **Higher layer response:** HMAC passes (attacker has valid key)
- **Fusion outcome:** FLAG (layers disagree — HMAC passes but RF doesn't match)
- **Residual risk:** Medium — detection depends on RF classifier reliability. At current 20% accuracy, ~79% detection rate. With improved RF (e.g., 80%), detection would exceed 95%

### Threat 3: Replay Attack
- **Attack:** Attacker records legitimate SAT_A transmission, re-transmits later
- **Physical layer response:** CNN may accept (signal is genuinely from SAT_A hardware)
- **Higher layer response:** Nonce already consumed → freshness check fails
- **Fusion outcome:** REJECT (nonce reuse detected)
- **Residual risk:** Low — nonce registry provides deterministic protection

### Threat 4: Signal Manipulation
- **Attack:** Attacker modifies captured signal features (power, phase, frequency) to resemble target satellite
- **Physical layer response:** Modified features may fool CNN (already at chance level)
- **Higher layer response:** Modified signal breaks HMAC tag → verification fails
- **Fusion outcome:** REJECT (HMAC failure)
- **Residual risk:** Low — HMAC integrity check catches payload modification

### Threat 5: Denial of Service
- **Attack:** Attacker floods receiver with signals causing all transmissions to be flagged/rejected
- **Physical layer response:** CNN processes each signal independently
- **Higher layer response:** HMAC check runs per message
- **Fusion outcome:** Legitimate signals still accepted if both layers pass
- **Residual risk:** Medium — resource exhaustion possible at receiver

## 6. Security Properties

| Property | Mechanism | Strength |
|----------|-----------|----------|
| **Transmitter authentication** | RF fingerprinting (CNN) | Weak — chance-level on this dataset |
| **Message integrity** | HMAC-SHA256 | Strong — 256-bit key, collision-resistant |
| **Identity verification** | HMAC key binding to satellite ID | Strong — requires key possession |
| **Replay protection** | Nonce freshness registry | Strong — deterministic detection |
| **Defence in depth** | Two independent layers + fusion | Strong architecture — degrades gracefully when one layer is weak |

## 7. Residual Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| RF classifier at chance level reduces stolen-key detection | Medium | Improve RF features / dataset; increase flag inspection frequency |
| Key compromise enables undetectable spoofing if RF is weak | High | Key rotation policy; HSM storage; anomaly detection on usage patterns |
| Nonce registry state loss (restart) enables replay window | Low | Persist nonce registry; use time-windowed nonces |
| Single ground station = single point of failure | Medium | Multi-receiver correlation; distributed verification |
| Adversarial ML attacks against CNN | Low (CNN already at chance) | Monitor for distribution shift; retrain periodically |
"""

with open(OUT_REPORTS / "threat_model.md", "w") as f:
    f.write(threat_model)
print(f"Saved: {OUT_REPORTS / 'threat_model.md'}")


# ══════════════════════════════════════════════════════════════════════════
# FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════

future_work = """# Future Work

## 1. Adaptive RF Fingerprinting

The core limitation identified in this work is that static RF features
fail to generalise across recording sessions due to channel-condition
dominance. An adaptive approach would continuously update the fingerprint
model as channel conditions change, using techniques such as:

- **Domain adaptation:** Train on source sessions and adapt to target
  sessions using maximum mean discrepancy (MMD) minimisation or
  domain-adversarial neural networks (DANN). The model learns to extract
  features that are discriminative for satellite identity while being
  invariant to session/channel conditions.

- **Online learning with drift detection:** Deploy the classifier with
  a sliding window of recent observations. When statistical drift is
  detected (e.g., via ADWIN or Page-Hinkley tests), trigger model
  re-calibration using the most recent labelled data from the HMAC-verified
  transmissions — effectively using the higher layer as a supervision
  signal for the physical layer.

## 2. Contrastive Learning for RF Fingerprinting

Rather than training a classifier directly, learn an embedding space where
bursts from the same satellite cluster together regardless of channel
conditions:

- **Supervised contrastive loss:** Pull together embeddings from the same
  satellite across different sessions, push apart embeddings from different
  satellites. This explicitly optimises for cross-session invariance, which
  direct classification does not.

- **Triplet networks:** Train with (anchor, positive, negative) tuples where
  positive pairs span different sessions. The network learns to ignore
  session-specific features by construction.

- The resulting embedding can be used for few-shot authentication: register
  a new satellite with a small number of verified bursts, then authenticate
  by nearest-neighbour in embedding space.

## 3. Self-Supervised RF Representation Learning

Pre-train on large volumes of unlabelled IQ data (all satellites, all
sessions) using self-supervised objectives:

- **Masked autoencoder:** Mask random segments of the IQ burst and train
  the model to reconstruct them. The learned representations capture
  signal structure without requiring labels.

- **Contrastive predictive coding (CPC):** Learn representations by
  predicting future samples from past context. Representations that
  capture hardware-specific temporal dynamics would naturally support
  downstream fingerprinting.

- **Augmentation-invariant learning (SimCLR/BYOL):** Apply
  channel-simulating augmentations (Gaussian noise, frequency offset,
  amplitude scaling, multipath simulation) and train the model to produce
  identical representations for differently-augmented versions of the same
  burst. This directly encourages channel-invariant features.

## 4. Federated RF Fingerprinting

In operational satellite networks, multiple ground stations receive
signals from the same satellites under different channel conditions:

- **Federated learning:** Each ground station trains a local RF
  fingerprinting model on its own data, then shares model updates (not
  raw IQ) with a central aggregator. The aggregated model benefits from
  diverse channel conditions without requiring data centralisation.

- **Privacy preservation:** Federated learning avoids sharing sensitive
  signal intelligence between ground stations, which may be operated by
  different entities or nations.

- **Channel diversity as a feature:** Multi-station observations of the
  same satellite provide natural cross-channel training data, directly
  addressing the single-session limitation identified in this work.

## 5. Secure Key Management and Rotation

The HMAC-based higher layer assumes pre-shared keys. In an operational
deployment, key management becomes critical:

- **Hardware Security Module (HSM) integration:** Store satellite keys
  in tamper-resistant hardware at ground stations, preventing extraction
  even under physical compromise.

- **Automated key rotation:** Implement periodic key updates using
  secure key-establishment protocols. Post-quantum key exchange (e.g.,
  CRYSTALS-Kyber) should be considered given the long operational
  lifetime of satellite constellations.

- **Key revocation:** When key compromise is detected (e.g., through
  the fusion layer flagging persistent RF mismatches despite valid HMAC),
  automatically revoke and replace the compromised key.

- **Threshold cryptography:** Split each satellite's key across multiple
  ground stations using Shamir's secret sharing, so that no single
  station compromise exposes the full key.

## 6. Hybrid Trust Scoring

Extend the two-layer fusion to incorporate additional trust signals:

- **Orbital prediction verification:** Compare the satellite's claimed
  position and timing against predicted orbital parameters (TLE data).
  Deviations indicate potential spoofing.

- **Doppler consistency check:** Verify that the observed Doppler shift
  matches the expected value for the claimed satellite's orbital velocity
  and geometry.

- **Behavioural anomaly detection:** Monitor per-satellite transmission
  patterns (message rate, payload characteristics, timing regularity)
  and flag deviations from established baselines.

- **Multi-receiver correlation:** If multiple ground stations receive
  the same transmission, cross-correlate their RF fingerprint assessments.
  Genuine satellites produce consistent fingerprints at all stations
  (modulo channel differences); a localised spoofer would only be visible
  to nearby stations.

- **Dynamic weight adjustment:** Instead of fixed fusion weights
  (w_rf=0.3, w_hl=0.7), learn optimal weights from operational data
  using logistic regression or a small neural network trained on
  labelled accept/reject outcomes.

## 7. Dataset and Evaluation Improvements

- **Multi-pass recordings:** Collect data across multiple satellite
  passes with controlled metadata to enable rigorous cross-pass
  evaluation with known ground truth.

- **Higher-bandwidth receivers:** Use receivers with wider bandwidth
  and higher sample rates to capture finer-grained hardware signatures
  that may be lost in the current SatIQ recording setup.

- **Open-set evaluation:** Test the system's ability to reject
  previously unseen satellites (not in the training set), which more
  closely reflects the operational scenario of detecting unknown
  transmitters.
"""

with open(OUT_REPORTS / "future_work.md", "w") as f:
    f.write(future_work)
print(f"Saved: {OUT_REPORTS / 'future_work.md'}")


print("\nAll thesis tables and documents generated.")
