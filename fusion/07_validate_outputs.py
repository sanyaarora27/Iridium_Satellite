#!/usr/bin/env python3
"""Validate final fusion outputs before dissertation numbers are frozen."""
from __future__ import annotations

import pandas as pd

from _shared import FUSION_TABLES_DIR, SATELLITES

def main() -> None:
    evidence = pd.read_csv(FUSION_TABLES_DIR / "evidence.csv")
    cases = pd.read_csv(FUSION_TABLES_DIR / "higher_layer_attack_cases.csv")
    attack_summary = pd.read_csv(
        FUSION_TABLES_DIR / "higher_layer_attack_results.csv"
    )
    population = pd.read_csv(
        FUSION_TABLES_DIR / "fusion_population_results.csv"
    )
    layer = pd.read_csv(FUSION_TABLES_DIR / "layer_comparison.csv")

    n_messages = len(evidence)
    n_false_claims = len(SATELLITES) - 1

    print("=" * 72)
    print("FINAL FUSION OUTPUT VALIDATION")
    print("=" * 72)

    # 1. RF provenance.
    if "rf_n_estimators" not in evidence.columns:
        raise AssertionError("evidence.csv lacks rf_n_estimators provenance")
    if set(evidence["rf_n_estimators"].astype(int)) != {200}:
        raise AssertionError("Fusion evidence is not from the 200-tree RF baseline")
    if "rf_scaled" in evidence.columns:
        scaled_values = set(evidence["rf_scaled"].astype(str).str.lower())
        if scaled_values not in ({"false"}, {"0"}):
            raise AssertionError("RF fusion baseline should not be scaled")
    print("[PASS] RF provenance: 200 trees, primary baseline configuration")

    # 2. Higher-layer expected/observed checks.
    if not (attack_summary["security_test_pass_rate"] == 1.0).all():
        raise AssertionError("One or more higher-layer tests failed")
    print("[PASS] Higher-layer expected-vs-observed security tests")

    # 3. False-identity coverage.
    expected_identity_cases = n_messages * n_false_claims
    for scenario in ["ordinary_identity_spoof", "stolen_key"]:
        sub = cases[cases["scenario"] == scenario]
        if len(sub) != expected_identity_cases:
            raise AssertionError(
                f"{scenario}: expected {expected_identity_cases}, found {len(sub)}"
            )
        if (sub["true_satellite"] == sub["claimed_satellite"]).any():
            raise AssertionError(f"{scenario}: true identity used as false claim")
        per_message = sub.groupby("message_id")["claimed_satellite"].nunique()
        if not (per_message == n_false_claims).all():
            raise AssertionError(f"{scenario}: incomplete false-claim coverage")
    print(
        "[PASS] False-identity coverage: "
        f"{n_messages:,} messages x {n_false_claims} incorrect claims"
    )

    # 4. Fusion join did not lose/duplicate cases.
    if len(population) != len(cases):
        raise AssertionError(
            f"Population rows {len(population)} != attack cases {len(cases)}"
        )
    if population["case_id"].duplicated().any():
        raise AssertionError("Duplicate case_id values in fusion population")
    print("[PASS] Fusion population contains exactly one row per attack case")

    # 5. Policy consistency.
    def expected_decision(r):
        if not bool(r["hmac_pass"]):
            return "reject"
        if not bool(r["freshness_pass"]):
            return "reject"
        if bool(r["rf_match"]):
            return "accept"
        return "flag"

    expected = population.apply(expected_decision, axis=1)
    bad = population[population["decision"] != expected]
    if not bad.empty:
        raise AssertionError(f"Fusion-policy mismatches: {len(bad)}")
    print("[PASS] Every fusion decision follows the declared final policy")

    # 6. Per-scenario decision counts sum to n for all architectures.
    count_groups = [
        ("rf_only_accept_count", "rf_only_reject_count"),
        ("hmac_only_accept_count", "hmac_only_reject_count"),
        (
            "hmac_plus_freshness_accept_count",
            "hmac_plus_freshness_reject_count",
        ),
        (
            "full_fusion_accept_count",
            "full_fusion_flag_count",
            "full_fusion_reject_count",
        ),
    ]
    for cols in count_groups:
        if not (layer[list(cols)].sum(axis=1).astype(int) == layer["n"].astype(int)).all():
            raise AssertionError(f"Decision counts do not sum to n for {cols}")
    print("[PASS] Architecture decision counts sum exactly to each denominator")

    print("\nAll final fusion validation checks: PASS")
    print(f"RF evidence messages: {n_messages:,}")
    print(f"Total simulated fusion cases: {len(population):,}")
    print(f"Identity-spoof cases: {expected_identity_cases:,}")
    print(f"Stolen-key cases: {expected_identity_cases:,}")

if __name__ == "__main__":
    main()
