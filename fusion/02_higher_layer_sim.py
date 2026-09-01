#!/usr/bin/env python3
"""
fusion/02_higher_layer_sim.py

Generate the genuine simulated higher-layer baseline.

For each RF-evidence record:
- create a deterministic synthetic payload
- create a fresh TOTP nonce
- sign identity + nonce + payload with HMAC-SHA256
- verify HMAC
- verify freshness

Output:
    fusion/outputs/tables/higher_layer_verdicts.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _shared import (
    COLUMN_MAP,
    FUSION_TABLES_DIR,
    SIM_BASE_TIME,
    TOTP_INTERVAL_SECONDS,
    build_simulated_payload,
    compute_mac,
    derive_hmac_key,
    generate_nonce,
    higher_layer_decision,
    verify_freshness,
    verify_mac,
)


def build_baseline_verdicts(evidence: pd.DataFrame) -> pd.DataFrame:
    id_col = COLUMN_MAP["id"]
    true_col = COLUMN_MAP["true_sat"]
    claimed_col = COLUMN_MAP["claimed_sat"]

    seen_nonces: set[tuple[int, str]] = set()
    rows = []

    for position, (_, row) in enumerate(evidence.iterrows()):
        message_id = int(row[id_col])
        true_sat = int(row[true_col])
        claimed_sat = int(row[claimed_col])

        sim_time = SIM_BASE_TIME + position * TOTP_INTERVAL_SECONDS

        payload = build_simulated_payload(message_id, true_sat)
        nonce = generate_nonce(claimed_sat, at_time=sim_time)
        key = derive_hmac_key(claimed_sat)

        mac = compute_mac(
            key=key,
            message_id=message_id,
            claimed_satellite=claimed_sat,
            nonce=nonce,
            payload=payload,
        )

        hmac_pass = verify_mac(
            key=key,
            message_id=message_id,
            claimed_satellite=claimed_sat,
            nonce=nonce,
            payload=payload,
            mac=mac,
        )

        freshness_pass = verify_freshness(
            sat_id=claimed_sat,
            nonce=nonce,
            at_time=sim_time,
            seen_nonces=seen_nonces,
        )

        rows.append(
            {
                id_col: message_id,
                "sim_time": sim_time,
                "sim_payload": payload,
                "sim_nonce": nonce,
                "sim_mac": mac,
                "hmac_pass": bool(hmac_pass),
                "freshness_pass": bool(freshness_pass),
                "higher_layer_decision": higher_layer_decision(
                    hmac_pass,
                    freshness_pass,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=FUSION_TABLES_DIR / "evidence.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FUSION_TABLES_DIR / "higher_layer_verdicts.csv",
    )
    args = parser.parse_args()

    evidence = pd.read_csv(args.evidence)

    required = [
        COLUMN_MAP["id"],
        COLUMN_MAP["true_sat"],
        COLUMN_MAP["claimed_sat"],
    ]
    missing = [col for col in required if col not in evidence.columns]

    if missing:
        sys.exit(
            f"Evidence file is missing required column(s): {missing}\n"
            f"Available columns: {list(evidence.columns)}"
        )

    verdicts = build_baseline_verdicts(evidence)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    verdicts.to_csv(args.out, index=False)

    n = len(verdicts)
    hmac_ok = int(verdicts["hmac_pass"].sum())
    fresh_ok = int(verdicts["freshness_pass"].sum())

    print(f"Wrote {n} genuine higher-layer verdicts to: {args.out}")
    print(f"HMAC pass:      {hmac_ok}/{n} ({100 * hmac_ok / n:.1f}%)")
    print(f"Freshness pass: {fresh_ok}/{n} ({100 * fresh_ok / n:.1f}%)")

    if hmac_ok != n or fresh_ok != n:
        sys.exit(
            "Baseline failure detected. Genuine simulated traffic should pass "
            "both higher-layer checks."
        )


if __name__ == "__main__":
    main()
