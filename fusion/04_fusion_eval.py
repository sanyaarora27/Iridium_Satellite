#!/usr/bin/env python3
"""
fusion/04_fusion_eval.py

Final multi-layer fusion policy.

Policy:
    HMAC failure                              -> REJECT
    Freshness failure                         -> REJECT
    Higher layer PASS + RF match              -> ACCEPT
    Higher layer PASS + RF mismatch           -> FLAG

RF disagreement is not allowed to autonomously reject a cryptographically
valid, fresh message because the current RF model has limited reliability.

The real/simulated boundary is explicit:
- RF evidence comes from the real held-out Iridium waveform observations.
- HMAC/freshness fields and attack scenarios are simulated overlays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import (
    COLUMN_MAP,
    FUSION_FIGURES_DIR,
    FUSION_REPORTS_DIR,
    FUSION_TABLES_DIR,
)

ID = COLUMN_MAP["id"]
TRUE = COLUMN_MAP["true_sat"]
CLAIMED = COLUMN_MAP["claimed_sat"]
PRED = COLUMN_MAP["predicted_sat"]
CONF = COLUMN_MAP["confidence"]

def decide(
    rf_match: bool,
    hmac_pass: bool,
    freshness_pass: bool,
) -> str:
    if not hmac_pass:
        return "reject"
    if not freshness_pass:
        return "reject"
    if rf_match:
        return "accept"
    return "flag"

def decision_reason(
    rf_match: bool,
    hmac_pass: bool,
    freshness_pass: bool,
) -> str:
    if not hmac_pass:
        return "HMAC authentication failed"
    if not freshness_pass:
        return "Freshness/replay check failed"
    if not rf_match:
        return (
            "Higher-layer authentication passed but RF-derived physical-layer "
            "evidence disagreed with the claimed satellite"
        )
    return (
        "Higher-layer authentication passed and RF-derived physical-layer "
        "evidence agreed with the claimed satellite"
    )

def build_fusion_population(
    evidence: pd.DataFrame,
    attack_cases: pd.DataFrame,
) -> pd.DataFrame:
    required_evidence = [
        ID,
        TRUE,
        PRED,
        CONF,
        "model_name",
        "feature_set",
        "split_type",
        "random_seed",
    ]
    missing_evidence = [c for c in required_evidence if c not in evidence.columns]
    if missing_evidence:
        raise KeyError(f"Evidence missing required columns: {missing_evidence}")

    required_cases = [
        "scenario",
        ID,
        TRUE,
        CLAIMED,
        "hmac_pass",
        "freshness_pass",
    ]
    missing_cases = [c for c in required_cases if c not in attack_cases.columns]
    if missing_cases:
        raise KeyError(f"Attack cases missing required columns: {missing_cases}")

    provenance_optional = [
        "n_features",
        "test_fraction",
        "rf_n_estimators",
        "rf_scaled",
        "rf_test_accuracy",
    ]
    rf_cols = required_evidence + [
        c for c in provenance_optional if c in evidence.columns
    ]
    rf = evidence[rf_cols].copy()

    merged = attack_cases.merge(
        rf,
        on=[ID, TRUE],
        how="inner",
        validate="many_to_one",
    )

    if len(merged) != len(attack_cases):
        raise AssertionError(
            f"Fusion join lost rows: attack_cases={len(attack_cases)}, merged={len(merged)}"
        )

    rows = []

    for _, row in merged.iterrows():
        claimed_sat = int(row[CLAIMED])
        predicted_sat = int(row[PRED])

        rf_match = predicted_sat == claimed_sat
        hmac_pass = bool(row["hmac_pass"])
        freshness_pass = bool(row["freshness_pass"])

        decision = decide(
            rf_match=rf_match,
            hmac_pass=hmac_pass,
            freshness_pass=freshness_pass,
        )

        higher_layer_pass = hmac_pass and freshness_pass

        out = {
            "case_id": row.get(
                "case_id",
                f"{row['scenario']}:{int(row[ID])}:{claimed_sat}",
            ),
            "scenario": row["scenario"],
            ID: int(row[ID]),
            TRUE: int(row[TRUE]),
            CLAIMED: claimed_sat,
            PRED: predicted_sat,
            CONF: float(row[CONF]),
            "rf_match": bool(rf_match),
            "hmac_pass": hmac_pass,
            "freshness_pass": freshness_pass,
            "higher_layer_pass": bool(higher_layer_pass),
            "decision": decision,
            "decision_reason": decision_reason(
                rf_match=rf_match,
                hmac_pass=hmac_pass,
                freshness_pass=freshness_pass,
            ),
            "higher_layer_reject": not higher_layer_pass,
            "combined_reject": decision == "reject",
            "combined_flag": decision == "flag",
            "combined_accept": decision == "accept",
            "security_escalation": decision in {"flag", "reject"},
            # Carry RF provenance into final fusion output.
            "model_name": row["model_name"],
            "feature_set": row["feature_set"],
            "split_type": row["split_type"],
            "random_seed": int(row["random_seed"]),
        }

        for col in provenance_optional:
            if col in row.index:
                out[col] = row[col]

        rows.append(out)

    population = pd.DataFrame(rows)

    # Policy invariant: recompute expected decisions independently.
    expected = np.where(
        ~population["hmac_pass"].astype(bool),
        "reject",
        np.where(
            ~population["freshness_pass"].astype(bool),
            "reject",
            np.where(population["rf_match"].astype(bool), "accept", "flag"),
        ),
    )
    mismatch = population["decision"].to_numpy() != expected
    if mismatch.any():
        raise AssertionError(
            f"Fusion-policy inconsistency in {int(mismatch.sum())} row(s)."
        )

    return population

def summarise(population: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, group in population.groupby("scenario", sort=False):
        n = len(group)
        accept_count = int(group["decision"].eq("accept").sum())
        flag_count = int(group["decision"].eq("flag").sum())
        reject_count = int(group["decision"].eq("reject").sum())

        rows.append(
            {
                "scenario": scenario,
                "n": n,
                "n_unique_messages": int(group[ID].nunique()),
                "accept_count": accept_count,
                "accept_rate": accept_count / n,
                "flag_count": flag_count,
                "flag_rate": flag_count / n,
                "reject_count": reject_count,
                "reject_rate": reject_count / n,
                "rf_match_rate": group["rf_match"].mean(),
                "higher_layer_pass_rate": group["higher_layer_pass"].mean(),
                "higher_layer_reject_rate": group["higher_layer_reject"].mean(),
                "security_escalation_rate": group["security_escalation"].mean(),
            }
        )

    return pd.DataFrame(rows)

def build_decision_table(population: pd.DataFrame) -> pd.DataFrame:
    """Select representative computed rows for human-readable inspection."""
    preferred_scenarios = [
        "genuine",
        "wrong_key",
        "tampered_payload",
        "ordinary_identity_spoof",
        "stolen_key",
        "replay",
        "expired_freshness",
    ]

    rows = []

    for scenario in preferred_scenarios:
        subset = population[population["scenario"] == scenario]
        if subset.empty:
            continue

        if scenario == "genuine":
            preferred = subset[subset["decision"] == "accept"]
        elif scenario == "stolen_key":
            preferred = subset[subset["decision"] == "flag"]
        else:
            preferred = subset[subset["decision"] == "reject"]

        row = preferred.iloc[0] if not preferred.empty else subset.iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)

def write_text_summary(
    summary: pd.DataFrame,
    evidence: pd.DataFrame,
    out_path: Path,
) -> None:
    rf_accuracy = (
        evidence[PRED].astype(int) == evidence[TRUE].astype(int)
    ).mean()

    lines = [
        "MULTI-LAYER AUTHENTICATION FUSION SUMMARY",
        "=" * 60,
        "",
        "Fusion policy:",
        "- HMAC failure -> REJECT",
        "- Freshness failure -> REJECT",
        "- Higher layer passes + RF match -> ACCEPT",
        "- Higher layer passes + RF mismatch -> FLAG",
        "",
        (
            "RF mismatch is intentionally treated as an escalation signal "
            "rather than an autonomous rejection criterion."
        ),
        "",
        f"RF evidence-set accuracy: {100 * rf_accuracy:.2f}%",
        "",
    ]

    for _, row in summary.iterrows():
        lines.extend(
            [
                f"Scenario: {row['scenario']}",
                f"  cases: {int(row['n']):,}",
                f"  unique RF observations: {int(row['n_unique_messages']):,}",
                f"  higher-layer pass: {100 * row['higher_layer_pass_rate']:.1f}%",
                f"  RF match: {100 * row['rf_match_rate']:.1f}%",
                (
                    f"  ACCEPT: {int(row['accept_count']):,}/"
                    f"{int(row['n']):,} ({100 * row['accept_rate']:.1f}%)"
                ),
                (
                    f"  FLAG: {int(row['flag_count']):,}/"
                    f"{int(row['n']):,} ({100 * row['flag_rate']:.1f}%)"
                ),
                (
                    f"  REJECT: {int(row['reject_count']):,}/"
                    f"{int(row['n']):,} ({100 * row['reject_rate']:.1f}%)"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "INTERPRETATION",
            "-" * 60,
            "",
            (
                "The stolen-key scenario intentionally passes the simulated "
                "higher layer because the attacker is modelled as possessing "
                "a valid target credential."
            ),
            "",
            (
                "RF therefore supplies an independent RF-derived physical-layer "
                "consistency signal in that scenario. This is not proof of a "
                "hardware-unique fingerprint because the measured RF features "
                "are affected by channel and received-signal conditions."
            ),
            "",
            (
                "A FLAG is an escalation signal, not a verified attack detection "
                "and not an autonomous rejection."
            ),
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")

def make_figure(summary: pd.DataFrame, out_path: Path) -> None:
    """Simple full-fusion distribution; thesis comparison figures come from 06."""
    focus_scenarios = [
        "genuine",
        "ordinary_identity_spoof",
        "replay",
        "stolen_key",
    ]

    focus = summary[summary["scenario"].isin(focus_scenarios)].copy()
    if focus.empty:
        return

    focus["scenario"] = pd.Categorical(
        focus["scenario"], categories=focus_scenarios, ordered=True
    )
    focus = focus.sort_values("scenario")

    labels = {
        "genuine": "Genuine",
        "ordinary_identity_spoof": "Claimed-ID spoof",
        "replay": "Replay",
        "stolen_key": "Stolen key",
    }
    categories = [
        f"{labels[str(row['scenario'])]}\n(n={int(row['n']):,})"
        for _, row in focus.iterrows()
    ]

    x = np.arange(len(categories))
    width = 0.25
    accept = 100 * focus["accept_rate"].to_numpy()
    flag = 100 * focus["flag_rate"].to_numpy()
    reject = 100 * focus["reject_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_a = ax.bar(x - width, accept, width, label="ACCEPT")
    bars_f = ax.bar(x, flag, width, label="FLAG")
    bars_r = ax.bar(x + width, reject, width, label="REJECT")

    for bars, values in [(bars_a, accept), (bars_f, flag), (bars_r, reject)]:
        for bar, value in zip(bars, values):
            if value >= 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Decision rate (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Full-Fusion Decision Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--evidence",
        type=Path,
        default=FUSION_TABLES_DIR / "evidence.csv",
    )
    parser.add_argument(
        "--attack-cases",
        type=Path,
        default=FUSION_TABLES_DIR / "higher_layer_attack_cases.csv",
    )

    args = parser.parse_args()

    evidence = pd.read_csv(args.evidence)
    attack_cases = pd.read_csv(args.attack_cases)

    population = build_fusion_population(
        evidence=evidence,
        attack_cases=attack_cases,
    )
    summary = summarise(population)
    decision_table = build_decision_table(population)

    population_path = FUSION_TABLES_DIR / "fusion_population_results.csv"
    summary_path = FUSION_TABLES_DIR / "fusion_summary.csv"
    decision_path = FUSION_TABLES_DIR / "decision_table.csv"
    text_path = FUSION_REPORTS_DIR / "headline_summary.txt"
    figure_path = FUSION_FIGURES_DIR / "fusion_decision_distribution.png"

    population.to_csv(population_path, index=False)
    summary.to_csv(summary_path, index=False)
    decision_table.to_csv(decision_path, index=False)

    write_text_summary(summary, evidence, text_path)
    make_figure(summary, figure_path)

    display = summary.copy()
    percentage_columns = [
        "accept_rate",
        "flag_rate",
        "reject_rate",
        "rf_match_rate",
        "higher_layer_pass_rate",
        "higher_layer_reject_rate",
        "security_escalation_rate",
    ]
    for col in percentage_columns:
        display[col] = (100 * display[col]).round(1)

    print()
    print("=" * 72)
    print("MULTI-LAYER AUTHENTICATION FUSION")
    print("=" * 72)
    print()
    print("Policy:")
    print("  HMAC failure                    -> REJECT")
    print("  Freshness failure               -> REJECT")
    print("  Higher layer PASS + RF match    -> ACCEPT")
    print("  Higher layer PASS + RF mismatch -> FLAG")
    print()
    print("=== Fusion summary (%) ===")
    print(display.to_string(index=False))
    print()
    print(f"Population results: {population_path}")
    print(f"Scenario summary:   {summary_path}")
    print(f"Decision examples:  {decision_path}")
    print(f"Text summary:       {text_path}")
    print(f"Decision figure:    {figure_path}")

if __name__ == "__main__":
    main()
