# Final Fusion Cleanup — Patch Notes

This cleanup implements the supervisor-facing methodological corrections before
final dissertation numbers are frozen.

## Changes

1. `01_evidence_adapter.py`
   - Random Forest changed from 300 trees to the authoritative 200-tree baseline.
   - Removed unnecessary `StandardScaler` so the model configuration matches
     the primary classical baseline.
   - Added RF provenance fields (`rf_n_estimators`, `rf_scaled`).
   - Added optional consistency check against `classifier_comparison.csv`.

2. `03_higher_layer_eval.py`
   - HMAC/freshness outcomes are produced by the implemented verification
     functions rather than literal pass values where practical.
   - Ordinary identity spoof and stolen-key scenarios now test all four
     incorrect claimed identities per RF observation.
   - Added unique `case_id` and coverage validation.
   - Scenario summaries now report unique messages and cases per message.

3. `04_fusion_eval.py`
   - Final policy remains unchanged.
   - Added join/policy invariants and explicit decision counts.
   - Replaced hardware-specific wording with RF-derived physical-layer evidence.
   - Replaced the old headline figure with `fusion_decision_distribution.png`.

4. `05_layer_comparison.py`
   - RF-only is now a true standalone authentication baseline:
     RF match -> ACCEPT; RF mismatch -> REJECT.
   - HMAC-only, HMAC+freshness, and full-fusion counts are written explicitly.
   - Counts and rates are both saved, reducing ambiguity about denominators.

5. `06_supervisor_comparison.py`
   - Reads explicit counts rather than reconstructing counts from rounded rates.
   - Produces ground-truth-aware ACCEPT/FLAG/REJECT tables.
   - Compares observed full fusion with ideal behaviour.
   - Compares RF only vs HMAC only vs HMAC+freshness vs full fusion.

6. `07_validate_outputs.py`
   - New final validation script.
   - Checks 200-tree RF provenance, scenario coverage, one-row-per-case fusion,
     policy consistency, and decision-count denominators.

7. Folder cleanup
   - Removed `.DS_Store`, `__pycache__`, backup script, and stale generated
     outputs. Regenerate outputs locally using the full pipeline.

## Required rerun

From the project root:

```bash
python fusion/01_evidence_adapter.py
python fusion/02_higher_layer_sim.py
python fusion/03_higher_layer_eval.py
python fusion/04_fusion_eval.py
python fusion/05_layer_comparison.py
python fusion/06_supervisor_comparison.py
python fusion/07_validate_outputs.py
```

Do not freeze dissertation numbers until the final validation script reports:

```text
All final fusion validation checks: PASS
```
