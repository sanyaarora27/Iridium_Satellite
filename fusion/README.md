# Fusion Layer

This folder implements the dissertation's proof-of-concept multi-layer
satellite authentication framework.

## Scope

The fusion layer combines:

1. **Real physical-layer evidence**
   - RF fingerprint prediction from real Iridium IQ-derived features.
2. **Simulated higher-layer authentication**
   - HMAC-SHA256 for keyed message/integrity authentication.
3. **Simulated freshness protection**
   - TOTP plus explicit replay-state tracking.

It **does not** implement the real Iridium authentication protocol, real
satellite key provisioning, or production key management.

The RF layer is described as **RF-derived physical-layer consistency evidence**.
It is not treated as proof of a hardware-unique fingerprint because the measured
features are affected by channel and received-signal conditions.

## Execution order

Run from the repository root:

```bash
python fusion/01_evidence_adapter.py
python fusion/02_higher_layer_sim.py
python fusion/03_higher_layer_eval.py
python fusion/04_fusion_eval.py
python fusion/05_layer_comparison.py
python fusion/06_supervisor_comparison.py
python fusion/07_validate_outputs.py
```

## Scripts

### `01_evidence_adapter.py`

Creates `evidence.csv` using the same primary Random Forest configuration as the
classical physical-layer baseline:

- 28 v1 waveform features
- stratified 80/20 split
- random seed 42
- 200 trees
- no scaling

The output includes model provenance.

### `02_higher_layer_sim.py`

Creates genuine simulated higher-layer messages and verifies:

- HMAC-SHA256
- TOTP freshness
- replay state

### `03_higher_layer_eval.py`

Runs the higher-layer security test suite:

- genuine
- wrong key
- tampered claim
- tampered payload
- invalid MAC
- replay
- expired freshness
- ordinary identity spoof
- stolen key / cloned credential

HMAC and freshness outcomes are produced by the implemented verification
functions. Expected and observed results are compared automatically.

For the two headline false-identity scenarios, **every held-out RF observation
is paired with all four incorrect claimed identities**. If the evidence set
contains 1,233 messages, this produces 4,932 ordinary-spoof cases and 4,932
stolen-key cases.

### `04_fusion_eval.py`

Applies the final proposed policy:

```text
HMAC fail                              -> REJECT
Freshness fail                         -> REJECT
Higher layer pass + RF match           -> ACCEPT
Higher layer pass + RF mismatch        -> FLAG
```

A `FLAG` is an escalation signal, not a verified attack detection.

RF mismatch does not autonomously reject a valid higher-layer message because
the current RF model has limited reliability.

### `05_layer_comparison.py`

Compares four authentication configurations across the same scenario
populations:

```text
RF only:
    RF match    -> ACCEPT
    RF mismatch -> REJECT

HMAC only:
    HMAC pass   -> ACCEPT
    HMAC fail   -> REJECT

HMAC + freshness:
    both pass   -> ACCEPT
    either fail -> REJECT

Full fusion:
    higher-layer failure -> REJECT
    higher-layer pass + RF match -> ACCEPT
    higher-layer pass + RF mismatch -> FLAG
```

The RF-only baseline is intentionally a hard standalone-authentication baseline.
The proposed full fusion reduces RF authority from hard rejection to escalation.

### `06_supervisor_comparison.py`

Produces the thesis-facing comparison requested by the supervisor. It reports:

- ground truth for each scenario
- exact denominator `n`
- ACCEPT / FLAG / REJECT counts and percentages
- whether a decision is a correct acceptance, false alarm, false acceptance,
  escalation, or block
- RF-only vs HMAC-only vs HMAC+freshness vs full fusion
- observed full-fusion behaviour vs ideal behaviour under the current policy

### `07_validate_outputs.py`

Runs consistency checks over the generated outputs before numbers are frozen for
the dissertation.

## Main thesis-facing outputs

Tables:

- `outputs/tables/evidence.csv`
- `outputs/tables/higher_layer_attack_results.csv`
- `outputs/tables/fusion_summary.csv`
- `outputs/tables/layer_comparison.csv`
- `outputs/tables/scenario_decision_counts.csv`
- `outputs/tables/architecture_decision_comparison.csv`
- `outputs/tables/ideal_full_fusion.csv`

Figures:

- `outputs/figures/fusion_decision_distribution.png`
- `outputs/figures/full_fusion_observed_vs_ideal.png`
- `outputs/figures/architecture_comparison.png`

Reports:

- `outputs/reports/headline_summary.txt`
- `outputs/reports/layer_comparison.md`
- `outputs/reports/supervisor_fusion_comparison.md`

## Interpretation boundary

A stolen valid credential intentionally passes HMAC and freshness. The RF layer
can then provide an independent physical-layer consistency check. However,
because the current RF classifier is weak and heavily affected by channel
conditions, RF disagreement is treated as supplementary escalation evidence,
not autonomous proof of an attack.
