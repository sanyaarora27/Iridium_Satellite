#!/usr/bin/env python3
"""
fusion/02_higher_layer_sim.py

Adds a SIMULATED higher-authentication-layer overlay -- HMAC (Option B) and
TOTP freshness (Option C) -- on top of the real RF evidence produced by
01_evidence_adapter.py.

Iridium carries no message authentication in reality, so everything this
script adds is a simulated overlay, not a property of the real signal.
Simulated fields are prefixed `sim_`. This script only builds the GENUINE
baseline: for every message in evidence.csv, it plays the role of the true
sender (who holds the correct key for the satellite it truly is) and
confirms the crypto validates as expected. Attack variants -- a stolen key,
a replayed nonce -- are constructed separately in 03_fusion_eval.py, reusing
the same primitives from _shared.py.

Input:
    fusion/evidence.csv        (from 01_evidence_adapter.py)
        Required columns (see _shared.COLUMN_MAP -- edit there if your
        real file uses different names):
            global_index     - join key
            claimed_sat      - satellite identity the message asserts
            rf_predicted_sat - RF classifier's top prediction (unused here,
                                passed through for 03's convenience)
            rf_confidence     - RF classifier's confidence (unused here)

Output:
    fusion/higher_layer_verdicts.csv
        global_index, sim_nonce, sim_mac, hmac_pass, freshness_pass

Usage:
    python 02_higher_layer_sim.py
    python 02_higher_layer_sim.py --evidence fusion/evidence.csv \
                                   --out fusion/higher_layer_verdicts.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from _shared import (
    COLUMN_MAP,
    BASE_TIME,
    derive_hmac_key,
    compute_mac,
    verify_mac,
    generate_nonce,
    verify_freshness,
)


def build_baseline_verdicts(evidence: pd.DataFrame) -> pd.DataFrame:
    id_col = COLUMN_MAP["id"]
    claimed_col = COLUMN_MAP["claimed_sat"]

    seen_nonces: set = set()
    rows = []

    for i, row in evidence.iterrows():
        gidx = row[id_col]
        claimed_sat = row[claimed_col]

        # A monotonically increasing "clock" per row stands in for message
        # timestamp. Real deployment would use the message's own capture
        # time; here we just need strictly increasing values so every
        # genuine message gets a distinct, fresh TOTP window.
        t = BASE_TIME + int(i) * 30

        key = derive_hmac_key(claimed_sat)
        nonce = generate_nonce(claimed_sat, at_time=t)
        mac = compute_mac(key, gidx, claimed_sat, nonce)

        hmac_pass = verify_mac(key, gidx, claimed_sat, nonce, mac)
        freshness_pass = verify_freshness(claimed_sat, nonce, at_time=t, seen_nonces=seen_nonces)

        rows.append({
            id_col: gidx,
            "sim_nonce": nonce,
            "sim_mac": mac,
            "hmac_pass": hmac_pass,
            "freshness_pass": freshness_pass,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="fusion/evidence.csv")
    ap.add_argument("--out", default="fusion/higher_layer_verdicts.csv")
    args = ap.parse_args()

    evidence = pd.read_csv(args.evidence)
    required = [COLUMN_MAP["id"], COLUMN_MAP["claimed_sat"]]
    missing = [c for c in required if c not in evidence.columns]
    if missing:
        sys.exit(
            f"evidence.csv is missing required column(s): {missing}\n"
            f"Available columns: {list(evidence.columns)}\n"
            f"Edit COLUMN_MAP in _shared.py to match your file."
        )

    verdicts = build_baseline_verdicts(evidence)
    verdicts.to_csv(args.out, index=False)

    n = len(verdicts)
    hmac_ok = verdicts["hmac_pass"].sum()
    fresh_ok = verdicts["freshness_pass"].sum()
    print(f"Wrote {n} verdicts to {args.out}")
    print(f"  HMAC pass:      {hmac_ok}/{n} ({100*hmac_ok/n:.1f}%)")
    print(f"  Freshness pass: {fresh_ok}/{n} ({100*fresh_ok/n:.1f}%)")
    if hmac_ok != n or fresh_ok != n:
        print(
            "  NOTE: baseline should be ~100% pass for genuine traffic -- "
            "investigate any failures before proceeding to 03_fusion_eval.py."
        )


if __name__ == "__main__":
    main()
