#!/usr/bin/env python3
"""
fusion/05_layer_comparison.py

Compare four authentication configurations across the same scenario
populations:

1. RF only
   RF agrees with claim -> ACCEPT
   RF disagrees         -> REJECT

2. HMAC only
   HMAC passes          -> ACCEPT
   HMAC fails           -> REJECT

3. HMAC + freshness
   both pass            -> ACCEPT
   either fails         -> REJECT

4. Full fusion (proposed policy)
   HMAC/freshness fail  -> REJECT
   both pass + RF match -> ACCEPT
   both pass + RF mismatch -> FLAG

The RF-only rule is intentionally a hard standalone-authentication baseline.
The proposed full fusion does NOT give RF that same rejection authority because
the measured RF model is weak.

This is a control-behaviour comparison, not a claim that every rejection is a
true attack detection. Genuine-traffic false alarms are reported separately by
06_supervisor_comparison.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import FUSION_REPORTS_DIR, FUSION_TABLES_DIR

def _count(mask: pd.Series) -> int:
    return int(mask.sum())

def build_comparison(population: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, group in population.groupby("scenario", sort=False):
        n = len(group)

        # RF-only standalone authenticator.
        rf_accept = _count(group["rf_match"].astype(bool))
        rf_reject = n - rf_accept

        # HMAC-only.
        hmac_accept = _count(group["hmac_pass"].astype(bool))
        hmac_reject = n - hmac_accept

        # HMAC + freshness.
        hf_accept = _count(
            group["hmac_pass"].astype(bool)
            & group["freshness_pass"].astype(bool)
        )
        hf_reject = n - hf_accept

        # Full fusion.
        full_accept = _count(group["decision"].eq("accept"))
        full_flag = _count(group["decision"].eq("flag"))
        full_reject = _count(group["decision"].eq("reject"))

        if full_accept + full_flag + full_reject != n:
            raise AssertionError(
                f"Full-fusion decision counts do not sum to n for {scenario}"
            )

        rows.append(
            {
                "scenario": scenario,
                "n": n,
                "n_unique_messages": int(group["message_id"].nunique()),

                "rf_only_accept_count": rf_accept,
                "rf_only_accept_rate": rf_accept / n,
                "rf_only_reject_count": rf_reject,
                "rf_only_reject_rate": rf_reject / n,

                "hmac_only_accept_count": hmac_accept,
                "hmac_only_accept_rate": hmac_accept / n,
                "hmac_only_reject_count": hmac_reject,
                "hmac_only_reject_rate": hmac_reject / n,

                "hmac_plus_freshness_accept_count": hf_accept,
                "hmac_plus_freshness_accept_rate": hf_accept / n,
                "hmac_plus_freshness_reject_count": hf_reject,
                "hmac_plus_freshness_reject_rate": hf_reject / n,

                "full_fusion_accept_count": full_accept,
                "full_fusion_accept_rate": full_accept / n,
                "full_fusion_flag_count": full_flag,
                "full_fusion_flag_rate": full_flag / n,
                "full_fusion_reject_count": full_reject,
                "full_fusion_reject_rate": full_reject / n,
                "full_fusion_escalation_count": full_flag + full_reject,
                "full_fusion_escalation_rate": (full_flag + full_reject) / n,
            }
        )

    return pd.DataFrame(rows)

def to_markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    rate_cols = [c for c in display.columns if c.endswith("_rate")]
    for col in rate_cols:
        display[col] = display[col].map(lambda x: f"{100*x:.1f}%")
    return display.to_markdown(index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        type=Path,
        default=FUSION_TABLES_DIR / "fusion_population_results.csv",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=FUSION_TABLES_DIR / "layer_comparison.csv",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=FUSION_REPORTS_DIR / "layer_comparison.md",
    )
    args = parser.parse_args()

    population = pd.read_csv(args.population)
    comparison = build_comparison(population)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.csv_out, index=False)

    report = [
        "# Authentication architecture comparison",
        "",
        "The same scenario populations are evaluated under four decision rules.",
        "",
        "- **RF only:** RF match = ACCEPT; RF mismatch = REJECT.",
        "- **HMAC only:** HMAC pass = ACCEPT; HMAC fail = REJECT.",
        "- **HMAC + freshness:** both checks pass = ACCEPT; otherwise REJECT.",
        (
            "- **Full fusion:** deterministic higher-layer failure = REJECT; "
            "otherwise RF match = ACCEPT and RF mismatch = FLAG."
        ),
        "",
        (
            "The RF-only baseline is intentionally harsher than the proposed "
            "fusion policy. It shows why a weak RF classifier should not be "
            "given autonomous rejection authority."
        ),
        "",
        to_markdown_table(comparison),
        "",
        "## Interpretation",
        "",
        "- HMAC handles invalid-key and authenticated-content failures.",
        "- Freshness adds replay/staleness protection.",
        (
            "- A stolen valid credential passes the simulated higher layer; "
            "RF supplies independent RF-derived physical-layer consistency evidence."
        ),
        (
            "- FLAG means escalation for investigation, not verified attack detection."
        ),
    ]

    args.md_out.write_text("\n".join(report), encoding="utf-8")

    display = comparison.copy()
    for col in [c for c in display.columns if c.endswith("_rate")]:
        display[col] = (100 * display[col]).round(1)

    print("\n=== Authentication architecture comparison (%) ===")
    print(display.to_string(index=False))
    print(f"\nSaved: {args.csv_out}")
    print(f"Saved: {args.md_out}")

if __name__ == "__main__":
    main()
