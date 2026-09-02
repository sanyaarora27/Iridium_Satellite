# Future Work

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


Current:
RF → HMAC → freshness → deterministic fusion

Future:
RF → HMAC → freshness
           ↓
      PyCasbin policy layer
           ↓
      decision / escalation
           ↓
      optional LLM advisor