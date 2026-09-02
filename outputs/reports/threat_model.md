# Threat Model: Multi-Layer Satellite Authentication

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
