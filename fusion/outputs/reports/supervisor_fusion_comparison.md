# Fusion comparison with explicit ground truth

## How to read the decisions

For **genuine** traffic:
- ACCEPT = correct acceptance.
- FLAG = false flag / false escalation.
- REJECT = false rejection.

For an **attack**:
- ACCEPT = false acceptance; the attack passes this decision rule.
- FLAG = escalation for investigation; it is not proof of attack detection.
- REJECT = the case is blocked by that decision rule.

Every percentage is scenario-specific: decision percentage = decision count / number of cases in that scenario.

## Full-fusion decision counts

| Scenario | Ground truth | n | ACCEPT | FLAG | REJECT | Ideal full-fusion outcome |
|---|---|---:|---:|---:|---:|---|
| Genuine | GENUINE | 1,233 | 302 (24.5%) | 931 (75.5%) | 0 (0.0%) | ACCEPT |
| Claimed-ID spoof | ATTACK | 4,932 | 0 (0.0%) | 0 (0.0%) | 4,932 (100.0%) | REJECT |
| Replay | ATTACK | 1,233 | 0 (0.0%) | 0 (0.0%) | 1,233 (100.0%) | REJECT |
| Stolen key | ATTACK | 4,932 | 931 (18.9%) | 4,001 (81.1%) | 0 (0.0%) | FLAG |

## Architecture comparison

| Scenario | Architecture | n | ACCEPT | FLAG | REJECT |
|---|---|---:|---:|---:|---:|
| Genuine | RF only | 1,233 | 302 (24.5%) | 0 (0.0%) | 931 (75.5%) |
| Genuine | HMAC only | 1,233 | 1,233 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| Genuine | HMAC + freshness | 1,233 | 1,233 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| Genuine | Full fusion | 1,233 | 302 (24.5%) | 931 (75.5%) | 0 (0.0%) |
| Claimed-ID spoof | RF only | 4,932 | 931 (18.9%) | 0 (0.0%) | 4,001 (81.1%) |
| Claimed-ID spoof | HMAC only | 4,932 | 0 (0.0%) | 0 (0.0%) | 4,932 (100.0%) |
| Claimed-ID spoof | HMAC + freshness | 4,932 | 0 (0.0%) | 0 (0.0%) | 4,932 (100.0%) |
| Claimed-ID spoof | Full fusion | 4,932 | 0 (0.0%) | 0 (0.0%) | 4,932 (100.0%) |
| Replay | RF only | 1,233 | 302 (24.5%) | 0 (0.0%) | 931 (75.5%) |
| Replay | HMAC only | 1,233 | 1,233 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| Replay | HMAC + freshness | 1,233 | 0 (0.0%) | 0 (0.0%) | 1,233 (100.0%) |
| Replay | Full fusion | 1,233 | 0 (0.0%) | 0 (0.0%) | 1,233 (100.0%) |
| Stolen key | RF only | 4,932 | 931 (18.9%) | 0 (0.0%) | 4,001 (81.1%) |
| Stolen key | HMAC only | 4,932 | 4,932 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| Stolen key | HMAC + freshness | 4,932 | 4,932 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| Stolen key | Full fusion | 4,932 | 931 (18.9%) | 4,001 (81.1%) | 0 (0.0%) |

## Key interpretation

The full-fusion system correctly ACCEPTS 24.5% of genuine cases and falsely FLAGs 75.5%.

In the stolen-key scenario, 81.1% are FLAGGED and 18.9% are falsely ACCEPTED. The stolen-key flag rate exceeds the genuine flag rate by only 5.62 percentage points, so the current RF evidence does not reliably separate stolen-key cases from genuine traffic.

Claimed-ID spoofing is rejected by HMAC because the attacker does not possess the target identity's credential. RF is not required for that rejection.

Replay shows the distinct contribution of freshness: HMAC alone accepts the still-valid authenticated packet, while HMAC plus freshness rejects its reuse.

RF-only rejection of replay should not be described as replay detection: the RF decision distribution is the same as for the corresponding genuine observation because replay does not change the physical RF evidence.

The false-identity scenarios evaluate every held-out RF observation against all four incorrect claimed identities, so their denominators are larger than the genuine and replay denominators.