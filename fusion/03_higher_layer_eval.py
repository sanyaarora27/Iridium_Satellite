#!/usr/bin/env python3
"""
fusion/03_higher_layer_eval.py

Security test suite for the simulated higher layer.

Each scenario is constructed and evaluated with the implemented HMAC and/or
freshness verification functions. Expected and observed outcomes are recorded
explicitly.

For the two headline false-identity scenarios (ordinary identity spoof and
stolen key), every held-out RF observation is tested against all four incorrect
claimed satellite identities. With 1,233 RF test observations and five target
satellites, this produces 4,932 cases per false-identity scenario.

Outputs:
    fusion/outputs/tables/higher_layer_attack_cases.csv
    fusion/outputs/tables/higher_layer_attack_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _shared import (
    COLUMN_MAP,
    FUSION_TABLES_DIR,
    SATELLITES,
    TOTP_INTERVAL_SECONDS,
    TOTP_VALID_WINDOW,
    alternate_satellites,
    build_simulated_payload,
    choose_alternate_satellite,
    compute_mac,
    corrupt_mac,
    derive_hmac_key,
    generate_nonce,
    higher_layer_decision,
    verify_freshness,
    verify_mac,
)

ID = COLUMN_MAP["id"]
TRUE = COLUMN_MAP["true_sat"]
CLAIMED = COLUMN_MAP["claimed_sat"]

EXPECTED = {
    "genuine": (True, True, "pass"),
    "wrong_key": (False, True, "reject"),
    "tampered_claim": (False, True, "reject"),
    "tampered_payload": (False, True, "reject"),
    "invalid_mac": (False, True, "reject"),
    "replay": (True, False, "reject"),
    "expired_freshness": (True, False, "reject"),
    "ordinary_identity_spoof": (False, True, "reject"),
    "stolen_key": (True, True, "pass"),
}

def _fresh_once(sat_id: int, nonce: str, sim_time: int) -> bool:
    """Evaluate one independent freshness use with clean replay state."""
    seen: set[tuple[int, str]] = set()
    return verify_freshness(
        sat_id=sat_id,
        nonce=nonce,
        at_time=sim_time,
        seen_nonces=seen,
    )

def _record(
    scenario: str,
    message_id: int,
    true_sat: int,
    claimed_sat: int,
    hmac_pass: bool,
    freshness_pass: bool,
) -> dict:
    observed_decision = higher_layer_decision(hmac_pass, freshness_pass)
    exp_hmac, exp_fresh, exp_decision = EXPECTED[scenario]

    test_passed = (
        bool(hmac_pass) == exp_hmac
        and bool(freshness_pass) == exp_fresh
        and observed_decision == exp_decision
    )

    return {
        "case_id": f"{scenario}:{int(message_id)}:{int(claimed_sat)}",
        "scenario": scenario,
        ID: int(message_id),
        TRUE: int(true_sat),
        CLAIMED: int(claimed_sat),
        "is_false_identity": int(claimed_sat) != int(true_sat),
        "expected_hmac_pass": exp_hmac,
        "observed_hmac_pass": bool(hmac_pass),
        "expected_freshness_pass": exp_fresh,
        "observed_freshness_pass": bool(freshness_pass),
        "expected_higher_layer_decision": exp_decision,
        "observed_higher_layer_decision": observed_decision,
        # Compatibility aliases consumed by 04_fusion_eval.py
        "hmac_pass": bool(hmac_pass),
        "freshness_pass": bool(freshness_pass),
        "higher_layer_decision": observed_decision,
        "higher_layer_reject": observed_decision == "reject",
        "test_passed": bool(test_passed),
    }

def build_attack_cases(
    evidence: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    merged = evidence.merge(
        baseline,
        on=ID,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_baseline"),
    )

    rows: list[dict] = []

    for _, row in merged.iterrows():
        message_id = int(row[ID])
        true_sat = int(row[TRUE])
        genuine_claim = int(row[CLAIMED])
        false_claim = choose_alternate_satellite(true_sat)

        sim_time = int(row["sim_time"])
        payload = str(row["sim_payload"])
        nonce = str(row["sim_nonce"])
        genuine_mac = str(row["sim_mac"])
        genuine_key = derive_hmac_key(genuine_claim)

        hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
            genuine_mac,
        )
        fresh_ok = _fresh_once(genuine_claim, nonce, sim_time)
        rows.append(
            _record(
                "genuine",
                message_id,
                true_sat,
                genuine_claim,
                hmac_ok,
                fresh_ok,
            )
        )

        # claimed identity and valid freshness value remain genuine.
        wrong_signing_key = derive_hmac_key(false_claim)
        wrong_key_mac = compute_mac(
            wrong_signing_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
        )
        hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
            wrong_key_mac,
        )
        fresh_ok = _fresh_once(genuine_claim, nonce, sim_time)
        rows.append(
            _record(
                "wrong_key",
                message_id,
                true_sat,
                genuine_claim,
                hmac_ok,
                fresh_ok,
            )
        )

        # valid freshness token is supplied for the false claim so that the
        # HMAC integrity failure is isolated rather than conflated with expiry.
        # The attacker still cannot produce a valid MAC for the altered packet.
        false_receiver_key = derive_hmac_key(false_claim)
        false_nonce = generate_nonce(false_claim, sim_time)
        hmac_ok = verify_mac(
            false_receiver_key,
            message_id,
            false_claim,
            false_nonce,
            payload,
            genuine_mac,
        )
        fresh_ok = _fresh_once(false_claim, false_nonce, sim_time)
        rows.append(
            _record(
                "tampered_claim",
                message_id,
                true_sat,
                false_claim,
                hmac_ok,
                fresh_ok,
            )
        )

        # still-valid freshness value are retained.
        tampered_payload = payload + "|tampered"
        hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            tampered_payload,
            genuine_mac,
        )
        fresh_ok = _fresh_once(genuine_claim, nonce, sim_time)
        rows.append(
            _record(
                "tampered_payload",
                message_id,
                true_sat,
                genuine_claim,
                hmac_ok,
                fresh_ok,
            )
        )

        bad_mac = corrupt_mac(genuine_mac)
        hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
            bad_mac,
        )
        fresh_ok = _fresh_once(genuine_claim, nonce, sim_time)
        rows.append(
            _record(
                "invalid_mac",
                message_id,
                true_sat,
                genuine_claim,
                hmac_ok,
                fresh_ok,
            )
        )

        # HMAC remains valid; the second freshness verification must fail.
        replay_hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
            genuine_mac,
        )
        replay_seen: set[tuple[int, str]] = set()
        first_use = verify_freshness(
            genuine_claim,
            nonce,
            sim_time,
            replay_seen,
        )
        second_use = verify_freshness(
            genuine_claim,
            nonce,
            sim_time,
            replay_seen,
        )
        if not first_use:
            raise RuntimeError(
                f"First freshness use unexpectedly failed for {message_id}"
            )
        rows.append(
            _record(
                "replay",
                message_id,
                true_sat,
                genuine_claim,
                replay_hmac_ok,
                second_use,
            )
        )

        # outside the accepted TOTP window. HMAC stays valid.
        expired_hmac_ok = verify_mac(
            genuine_key,
            message_id,
            genuine_claim,
            nonce,
            payload,
            genuine_mac,
        )
        expiry_offset = (TOTP_VALID_WINDOW + 3) * TOTP_INTERVAL_SECONDS
        expired_time = sim_time + expiry_offset
        expired_seen: set[tuple[int, str]] = set()
        expired_ok = verify_freshness(
            genuine_claim,
            nonce,
            expired_time,
            expired_seen,
        )
        rows.append(
            _record(
                "expired_freshness",
                message_id,
                true_sat,
                genuine_claim,
                expired_hmac_ok,
                expired_ok,
            )
        )

        # Headline false-identity scenarios.
        # Every real RF observation is tested against ALL four incorrect
        # claimed identities, avoiding dependence on one arbitrary target.
        for target_claim in alternate_satellites(true_sat):
            target_nonce = generate_nonce(target_claim, sim_time)
            target_payload = build_simulated_payload(message_id, true_sat)
            target_key = derive_hmac_key(target_claim)

            # Ordinary identity spoof: attacker has only the true source's own
            # key, not the target identity's credential.
            attacker_own_key = derive_hmac_key(true_sat)
            spoof_mac = compute_mac(
                attacker_own_key,
                message_id,
                target_claim,
                target_nonce,
                target_payload,
            )
            spoof_hmac_ok = verify_mac(
                target_key,
                message_id,
                target_claim,
                target_nonce,
                target_payload,
                spoof_mac,
            )
            spoof_fresh_ok = _fresh_once(
                target_claim,
                target_nonce,
                sim_time,
            )
            rows.append(
                _record(
                    "ordinary_identity_spoof",
                    message_id,
                    true_sat,
                    target_claim,
                    spoof_hmac_ok,
                    spoof_fresh_ok,
                )
            )

            # Stolen key / cloned credential: attacker is assumed to possess
            # the target identity's valid credential, so both deterministic
            # higher-layer checks should pass by construction.
            stolen_mac = compute_mac(
                target_key,
                message_id,
                target_claim,
                target_nonce,
                target_payload,
            )
            stolen_hmac_ok = verify_mac(
                target_key,
                message_id,
                target_claim,
                target_nonce,
                target_payload,
                stolen_mac,
            )
            stolen_fresh_ok = _fresh_once(
                target_claim,
                target_nonce,
                sim_time,
            )
            rows.append(
                _record(
                    "stolen_key",
                    message_id,
                    true_sat,
                    target_claim,
                    stolen_hmac_ok,
                    stolen_fresh_ok,
                )
            )

    cases = pd.DataFrame(rows)

    if cases["case_id"].duplicated().any():
        duplicates = cases.loc[cases["case_id"].duplicated(), "case_id"].head()
        raise AssertionError(f"Duplicate attack case IDs found: {duplicates.tolist()}")

    return cases

def summarise_attack_cases(cases: pd.DataFrame) -> pd.DataFrame:
    summaries = []

    for scenario, group in cases.groupby("scenario", sort=False):
        n_unique_messages = int(group[ID].nunique())
        summaries.append(
            {
                "scenario": scenario,
                "n": len(group),
                "n_unique_messages": n_unique_messages,
                "cases_per_message": len(group) / n_unique_messages,
                "n_unique_claimed_satellites": int(group[CLAIMED].nunique()),
                "hmac_pass_rate": group["hmac_pass"].mean(),
                "freshness_pass_rate": group["freshness_pass"].mean(),
                "higher_layer_pass_rate": (
                    group["higher_layer_decision"].eq("pass").mean()
                ),
                "higher_layer_reject_rate": (
                    group["higher_layer_decision"].eq("reject").mean()
                ),
                "security_test_pass_rate": group["test_passed"].mean(),
            }
        )

    return pd.DataFrame(summaries)

def validate_security_tests(summary: pd.DataFrame) -> None:
    failures = summary[summary["security_test_pass_rate"] != 1.0]

    if not failures.empty:
        raise AssertionError(
            "One or more higher-layer security tests did not match the "
            "expected outcome:\n"
            + failures.to_string(index=False)
        )

    expected_scenarios = set(EXPECTED)
    observed_scenarios = set(summary["scenario"])
    missing = expected_scenarios - observed_scenarios

    if missing:
        raise AssertionError(
            f"Expected higher-layer scenarios were not generated: {sorted(missing)}"
        )

def validate_false_identity_coverage(
    cases: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    """Require all four incorrect claims for each message in headline attacks."""
    expected_per_message = len(SATELLITES) - 1
    expected_total = len(evidence) * expected_per_message

    for scenario in ["ordinary_identity_spoof", "stolen_key"]:
        sub = cases[cases["scenario"] == scenario]

        if len(sub) != expected_total:
            raise AssertionError(
                f"{scenario}: expected {expected_total} cases, found {len(sub)}"
            )

        if (sub[TRUE].astype(int) == sub[CLAIMED].astype(int)).any():
            raise AssertionError(
                f"{scenario}: a false-identity case uses the true identity as claim"
            )

        counts = sub.groupby(ID)[CLAIMED].nunique()
        if not (counts == expected_per_message).all():
            bad = counts[counts != expected_per_message].head()
            raise AssertionError(
                f"{scenario}: not every message has all four false claims:\n{bad}"
            )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--evidence",
        type=Path,
        default=FUSION_TABLES_DIR / "evidence.csv",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=FUSION_TABLES_DIR / "higher_layer_verdicts.csv",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        default=FUSION_TABLES_DIR / "higher_layer_attack_cases.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=FUSION_TABLES_DIR / "higher_layer_attack_results.csv",
    )

    args = parser.parse_args()

    evidence = pd.read_csv(args.evidence)
    baseline = pd.read_csv(
        args.baseline,
        dtype={"sim_nonce": str, "sim_mac": str},
    )

    cases = build_attack_cases(evidence, baseline)
    summary = summarise_attack_cases(cases)

    validate_security_tests(summary)
    validate_false_identity_coverage(cases, evidence)

    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.cases_out, index=False)
    summary.to_csv(args.summary_out, index=False)

    display = summary.copy()
    for col in [
        "hmac_pass_rate",
        "freshness_pass_rate",
        "higher_layer_pass_rate",
        "higher_layer_reject_rate",
        "security_test_pass_rate",
    ]:
        display[col] = (100 * display[col]).round(1)

    print(f"Wrote attack cases to: {args.cases_out}")
    print(f"Wrote attack summary to: {args.summary_out}")
    print("\n=== Higher-layer security test suite (%) ===")
    print(display.to_string(index=False))
    print("\nAll expected-vs-observed security tests: PASS")
    print(
        "False-identity coverage: PASS "
        f"({len(evidence):,} messages x {len(SATELLITES)-1} incorrect claims = "
        f"{len(evidence)*(len(SATELLITES)-1):,} cases per headline identity scenario)"
    )

if __name__ == "__main__":
    main()
