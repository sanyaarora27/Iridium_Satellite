# Multi-Layer Satellite Authentication Prototype

This repository contains the source code and experimental outputs for an MSc
Cyber Security dissertation investigating satellite authentication using
physical-layer RF fingerprinting together with simulated higher-layer
authentication and freshness controls.

## Project Structure

- `scripts/` - RF fingerprinting, feature extraction, machine-learning
  evaluation and diagnostic experiments.
- `fusion/` - higher-layer authentication simulation, fusion policy and
  security-scenario evaluation.
- `tests/` - automated tests for evaluation methodology, CNN utilities,
  open-set evaluation and fusion logic.
- `outputs/` - generated tables, figures and reports used during evaluation.

## Dataset

The raw Iridium I/Q dataset is not included in this archive because of its
size. The experimental pipeline expects the required dataset files under:

    data/raw/

The experiments use observations from five Iridium satellites:
51, 85, 87, 92 and 109.

## Environment

Python 3.10.20

Install dependencies with:

    pip install -r requirements.txt

## Main Experimental Scripts

Classical RF fingerprinting:

    python scripts/05_train_classifiers.py

Pass-aware evaluation:

    python scripts/06_pass_aware_evaluation.py

Authentication metrics:

    python scripts/20_authentication_metrics.py

Open-set and cross-domain evaluation:

    python scripts/21_openset_and_domain.py

Final Chapter 5 validation:

    python scripts/33_ch5_final_validation.py

Fusion evaluation:

    python fusion/04_fusion_eval.py

Fusion output validation:

    python fusion/07_validate_outputs.py

## Automated Tests

Run:

    python -m pytest -q

The submitted version passes 26 automated tests.

## Important Scope Note

The higher-layer HMAC and freshness mechanisms are simulated prototype
controls and do not implement the operational Iridium authentication protocol.
RF fingerprint evidence is treated as supplementary evidence rather than an
autonomous authentication mechanism.
