#!/usr/bin/env python3
"""
fusion/06_supervisor_comparison.py

Supervisor/thesis reporting for the fusion experiment.

This script does not change the authentication algorithm. It converts the
computed layer comparison into explicit counts, denominators, ground-truth
meaning, ideal behaviour, and architecture-to-architecture comparisons.

Headline scenarios:
- genuine
- ordinary_identity_spoof
- replay
- stolen_key
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _shared import FUSION_FIGURES_DIR, FUSION_REPORTS_DIR, FUSION_TABLES_DIR

INPUT = FUSION_TABLES_DIR / "layer_comparison.csv"

SCENARIOS = [
    "genuine",
    "ordinary_identity_spoof",
    "replay",
    "stolen_key",
]

DISPLAY_NAMES = {
    "genuine": "Genuine",
    "ordinary_identity_spoof": "Claimed-ID spoof",
    "replay": "Replay",
    "stolen_key": "Stolen key",
}

GROUND_TRUTH = {
    "genuine": "GENUINE",
    "ordinary_identity_spoof": "ATTACK",
    "replay": "ATTACK",
    "stolen_key": "ATTACK",
}

# Ideal behaviour under the CURRENT proposed policy.
# A stolen-key case is ideally FLAGGED rather than rejected because the RF
# layer is deliberately not granted autonomous rejection authority.
IDEAL_DECISION = {
    "genuine": "ACCEPT",
    "ordinary_identity_spoof": "REJECT",
    "replay": "REJECT",
    "stolen_key": "FLAG",
}

ARCHITECTURES = [
    "RF only",
    "HMAC only",
    "HMAC + freshness",
    "Full fusion",
]

def _row_from_counts(
    scenario: str,
    architecture: str,
    n: int,
    accept: int,
    flag: int,
    reject: int,
) -> dict:
    if accept + flag + reject != n:
        raise AssertionError(
            f"{scenario}/{architecture}: ACCEPT+FLAG+REJECT != n"
        )

    truth = GROUND_TRUTH[scenario]

    row = {
        "scenario": scenario,
        "scenario_label": DISPLAY_NAMES[scenario],
        "ground_truth": truth,
        "architecture": architecture,
        "n_cases": n,
        "accept_count": accept,
        "accept_rate": accept / n,
        "flag_count": flag,
        "flag_rate": flag / n,
        "reject_count": reject,
        "reject_rate": reject / n,
    }

    if truth == "GENUINE":
        row.update(
            {
                "correct_accept_count": accept,
                "correct_accept_rate": accept / n,
                "false_alarm_count": flag + reject,
                "false_alarm_rate": (flag + reject) / n,
                "false_accept_count": 0,
                "false_accept_rate": 0.0,
                "attack_flag_count": 0,
                "attack_flag_rate": 0.0,
                "attack_block_count": 0,
                "attack_block_rate": 0.0,
            }
        )
    else:
        row.update(
            {
                "correct_accept_count": 0,
                "correct_accept_rate": 0.0,
                "false_alarm_count": 0,
                "false_alarm_rate": 0.0,
                "false_accept_count": accept,
                "false_accept_rate": accept / n,
                "attack_flag_count": flag,
                "attack_flag_rate": flag / n,
                "attack_block_count": reject,
                "attack_block_rate": reject / n,
            }
        )

    return row

def build_architecture_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    required = {
        "scenario",
        "n",
        "rf_only_accept_count",
        "rf_only_reject_count",
        "hmac_only_accept_count",
        "hmac_only_reject_count",
        "hmac_plus_freshness_accept_count",
        "hmac_plus_freshness_reject_count",
        "full_fusion_accept_count",
        "full_fusion_flag_count",
        "full_fusion_reject_count",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(
            "layer_comparison.csv is missing the new explicit-count columns: "
            f"{missing}. Run fusion/05_layer_comparison.py first."
        )

    for scenario in SCENARIOS:
        src = raw[raw["scenario"] == scenario]
        if src.empty:
            raise ValueError(f"Missing scenario: {scenario}")

        r = src.iloc[0]
        n = int(r["n"])

        rows.append(
            _row_from_counts(
                scenario,
                "RF only",
                n,
                int(r["rf_only_accept_count"]),
                0,
                int(r["rf_only_reject_count"]),
            )
        )
        rows.append(
            _row_from_counts(
                scenario,
                "HMAC only",
                n,
                int(r["hmac_only_accept_count"]),
                0,
                int(r["hmac_only_reject_count"]),
            )
        )
        rows.append(
            _row_from_counts(
                scenario,
                "HMAC + freshness",
                n,
                int(r["hmac_plus_freshness_accept_count"]),
                0,
                int(r["hmac_plus_freshness_reject_count"]),
            )
        )
        rows.append(
            _row_from_counts(
                scenario,
                "Full fusion",
                n,
                int(r["full_fusion_accept_count"]),
                int(r["full_fusion_flag_count"]),
                int(r["full_fusion_reject_count"]),
            )
        )

    return pd.DataFrame(rows)

def build_full_fusion_table(comparison: pd.DataFrame) -> pd.DataFrame:
    full = comparison[comparison["architecture"] == "Full fusion"].copy()
    rows = []

    for _, r in full.iterrows():
        scenario = r["scenario"]
        truth = r["ground_truth"]
        n = int(r["n_cases"])

        for decision in ["ACCEPT", "FLAG", "REJECT"]:
            key = decision.lower()
            count = int(r[f"{key}_count"])
            rate = float(r[f"{key}_rate"])

            if truth == "GENUINE":
                meaning = {
                    "ACCEPT": "Correct acceptance",
                    "FLAG": "False flag / false escalation",
                    "REJECT": "False rejection",
                }[decision]
            else:
                meaning = {
                    "ACCEPT": "False acceptance / attack passed",
                    "FLAG": "Attack escalated for investigation",
                    "REJECT": "Attack blocked by decision rule",
                }[decision]

            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": DISPLAY_NAMES[scenario],
                    "ground_truth": truth,
                    "n_cases": n,
                    "decision": decision,
                    "count": count,
                    "percentage": rate,
                    "meaning": meaning,
                }
            )

    return pd.DataFrame(rows)

def build_ideal_table(comparison: pd.DataFrame) -> pd.DataFrame:
    full = comparison[comparison["architecture"] == "Full fusion"]
    rows = []

    for _, r in full.iterrows():
        scenario = r["scenario"]
        n = int(r["n_cases"])
        ideal = IDEAL_DECISION[scenario]

        rows.append(
            {
                "scenario": scenario,
                "scenario_label": DISPLAY_NAMES[scenario],
                "ground_truth": GROUND_TRUTH[scenario],
                "n_cases": n,
                "ideal_decision": ideal,
                "ideal_accept_count": n if ideal == "ACCEPT" else 0,
                "ideal_flag_count": n if ideal == "FLAG" else 0,
                "ideal_reject_count": n if ideal == "REJECT" else 0,
            }
        )

    return pd.DataFrame(rows)

def plot_observed_vs_ideal(
    comparison: pd.DataFrame,
    ideal: pd.DataFrame,
    path: Path,
) -> None:
    full = (
        comparison[comparison["architecture"] == "Full fusion"]
        .set_index("scenario")
        .loc[SCENARIOS]
    )
    ideal = ideal.set_index("scenario").loc[SCENARIOS]

    labels = [DISPLAY_NAMES[s] for s in SCENARIOS]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13, 7))

    actual_bottom = np.zeros(len(labels))
    ideal_bottom = np.zeros(len(labels))

    for decision in ["ACCEPT", "FLAG", "REJECT"]:
        key = decision.lower()
        observed = full[f"{key}_rate"].to_numpy()
        target = np.array(
            [
                ideal.loc[s, f"ideal_{key}_count"] / ideal.loc[s, "n_cases"]
                for s in SCENARIOS
            ]
        )

        bars = ax.bar(
            x - width / 2,
            observed,
            width,
            bottom=actual_bottom,
            label=f"Observed {decision}",
        )
        ax.bar(
            x + width / 2,
            target,
            width,
            bottom=ideal_bottom,
            alpha=0.30,
            hatch="//",
            edgecolor="black",
        )

        for idx, bar in enumerate(bars):
            if observed[idx] >= 0.03:
                count = int(full.iloc[idx][f"{key}_count"])
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    actual_bottom[idx] + observed[idx] / 2,
                    f"{count:,}\n{observed[idx]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        actual_bottom += observed
        ideal_bottom += target

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proportion of cases")
    ax.set_title("Full Fusion: Observed Decisions vs Ideal Behaviour")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{label}\n(n={int(full.loc[s, 'n_cases']):,})"
            for label, s in zip(labels, SCENARIOS)
        ]
    )
    ax.text(
        0.99,
        0.02,
        "Solid = observed | Hatched/faded = ideal under current policy",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_architecture_comparison(
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()

    for ax, scenario in zip(axes, SCENARIOS):
        sub = (
            comparison[comparison["scenario"] == scenario]
            .set_index("architecture")
            .loc[ARCHITECTURES]
        )

        x = np.arange(len(ARCHITECTURES))
        bottom = np.zeros(len(ARCHITECTURES))

        for decision in ["ACCEPT", "FLAG", "REJECT"]:
            key = decision.lower()
            vals = sub[f"{key}_rate"].to_numpy()
            bars = ax.bar(x, vals, bottom=bottom, label=decision)

            for i, bar in enumerate(bars):
                if vals[i] >= 0.08:
                    count = int(sub.iloc[i][f"{key}_count"])
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[i] + vals[i] / 2,
                        f"{count:,}\n{vals[i]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            bottom += vals

        truth = GROUND_TRUTH[scenario]
        n = int(sub.iloc[0]["n_cases"])
        ax.set_title(
            f"{DISPLAY_NAMES[scenario]} | Ground truth: {truth} | n={n:,}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ARCHITECTURES, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Proportion of cases")
    axes[2].set_ylabel("Proportion of cases")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(
        "Authentication Architecture Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def write_report(
    comparison: pd.DataFrame,
    full_table: pd.DataFrame,
    ideal: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Fusion comparison with explicit ground truth",
        "",
        "## How to read the decisions",
        "",
        "For **genuine** traffic:",
        "- ACCEPT = correct acceptance.",
        "- FLAG = false flag / false escalation.",
        "- REJECT = false rejection.",
        "",
        "For an **attack**:",
        "- ACCEPT = false acceptance; the attack passes this decision rule.",
        "- FLAG = escalation for investigation; it is not proof of attack detection.",
        "- REJECT = the case is blocked by that decision rule.",
        "",
        (
            "Every percentage is scenario-specific: decision percentage = "
            "decision count / number of cases in that scenario."
        ),
        "",
        "## Full-fusion decision counts",
        "",
        "| Scenario | Ground truth | n | ACCEPT | FLAG | REJECT | Ideal full-fusion outcome |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for scenario in SCENARIOS:
        sub = full_table[full_table["scenario"] == scenario]
        n = int(sub.iloc[0]["n_cases"])
        vals = {}
        for decision in ["ACCEPT", "FLAG", "REJECT"]:
            r = sub[sub["decision"] == decision].iloc[0]
            vals[decision] = f"{int(r['count']):,} ({r['percentage']:.1%})"

        lines.append(
            f"| {DISPLAY_NAMES[scenario]} | {GROUND_TRUTH[scenario]} | {n:,} "
            f"| {vals['ACCEPT']} | {vals['FLAG']} | {vals['REJECT']} "
            f"| {IDEAL_DECISION[scenario]} |"
        )

    lines += [
        "",
        "## Architecture comparison",
        "",
        "| Scenario | Architecture | n | ACCEPT | FLAG | REJECT |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for scenario in SCENARIOS:
        sub = comparison[comparison["scenario"] == scenario]
        for _, r in sub.iterrows():
            lines.append(
                f"| {DISPLAY_NAMES[scenario]} | {r['architecture']} "
                f"| {int(r['n_cases']):,} "
                f"| {int(r['accept_count']):,} ({r['accept_rate']:.1%}) "
                f"| {int(r['flag_count']):,} ({r['flag_rate']:.1%}) "
                f"| {int(r['reject_count']):,} ({r['reject_rate']:.1%}) |"
            )

    full = (
        comparison[comparison["architecture"] == "Full fusion"]
        .set_index("scenario")
        .loc[SCENARIOS]
    )

    genuine_flag = float(full.loc["genuine", "flag_rate"])
    stolen_flag = float(full.loc["stolen_key", "flag_rate"])
    stolen_false_accept = float(full.loc["stolen_key", "false_accept_rate"])
    difference_pp = 100 * (stolen_flag - genuine_flag)

    lines += [
        "",
        "## Key interpretation",
        "",
        (
            f"The full-fusion system correctly ACCEPTS "
            f"{full.loc['genuine', 'accept_rate']:.1%} of genuine cases and "
            f"falsely FLAGs {genuine_flag:.1%}."
        ),
        "",
        (
            f"In the stolen-key scenario, {stolen_flag:.1%} are FLAGGED and "
            f"{stolen_false_accept:.1%} are falsely ACCEPTED. The stolen-key "
            f"flag rate exceeds the genuine flag rate by only {difference_pp:.2f} "
            f"percentage points, so the current RF evidence does not reliably "
            f"separate stolen-key cases from genuine traffic."
        ),
        "",
        (
            "Claimed-ID spoofing is rejected by HMAC because the attacker does "
            "not possess the target identity's credential. RF is not required "
            "for that rejection."
        ),
        "",
        (
            "Replay shows the distinct contribution of freshness: HMAC alone "
            "accepts the still-valid authenticated packet, while HMAC plus "
            "freshness rejects its reuse."
        ),
        "",
        (
            "RF-only rejection of replay should not be described as replay "
            "detection: the RF decision distribution is the same as for the "
            "corresponding genuine observation because replay does not change "
            "the physical RF evidence."
        ),
        "",
        (
            "The false-identity scenarios evaluate every held-out RF observation "
            "against all four incorrect claimed identities, so their denominators "
            "are larger than the genuine and replay denominators."
        ),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    print("=" * 80)
    print("SUPERVISOR / THESIS FUSION COMPARISON")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run fusion/05_layer_comparison.py first."
        )

    raw = pd.read_csv(INPUT)
    missing_scenarios = [s for s in SCENARIOS if s not in set(raw["scenario"])]
    if missing_scenarios:
        raise ValueError(f"Required scenarios missing: {missing_scenarios}")

    comparison = build_architecture_comparison(raw)
    full_table = build_full_fusion_table(comparison)
    ideal = build_ideal_table(comparison)

    comparison_path = FUSION_TABLES_DIR / "architecture_decision_comparison.csv"
    full_path = FUSION_TABLES_DIR / "scenario_decision_counts.csv"
    ideal_path = FUSION_TABLES_DIR / "ideal_full_fusion.csv"

    comparison.to_csv(comparison_path, index=False)
    full_table.to_csv(full_path, index=False)
    ideal.to_csv(ideal_path, index=False)

    observed_figure = FUSION_FIGURES_DIR / "full_fusion_observed_vs_ideal.png"
    architecture_figure = FUSION_FIGURES_DIR / "architecture_comparison.png"

    plot_observed_vs_ideal(comparison, ideal, observed_figure)
    plot_architecture_comparison(comparison, architecture_figure)

    report_path = FUSION_REPORTS_DIR / "supervisor_fusion_comparison.md"
    write_report(comparison, full_table, ideal, report_path)

    full = comparison[comparison["architecture"] == "Full fusion"][
        [
            "scenario_label",
            "ground_truth",
            "n_cases",
            "accept_count",
            "accept_rate",
            "flag_count",
            "flag_rate",
            "reject_count",
            "reject_rate",
        ]
    ]

    print("\nFULL FUSION - COUNTS AND DENOMINATORS")
    print("-" * 80)
    print(full.to_string(index=False))

    print("\nARCHITECTURE COMPARISON")
    print("-" * 80)
    display = comparison[
        [
            "scenario_label",
            "architecture",
            "n_cases",
            "accept_count",
            "accept_rate",
            "flag_count",
            "flag_rate",
            "reject_count",
            "reject_rate",
        ]
    ]
    print(display.to_string(index=False))

    print("\nSaved:")
    print(f"  {comparison_path}")
    print(f"  {full_path}")
    print(f"  {ideal_path}")
    print(f"  {observed_figure}")
    print(f"  {architecture_figure}")
    print(f"  {report_path}")

    print("\nInterpretation reminder:")
    print("  GENUINE + ACCEPT = correct acceptance")
    print("  GENUINE + FLAG/REJECT = false alarm/escalation")
    print("  ATTACK  + ACCEPT = false acceptance / attack passed")
    print("  ATTACK  + FLAG = escalation, not proven attack detection")
    print("  ATTACK  + REJECT = blocked by the decision rule")

if __name__ == "__main__":
    main()
