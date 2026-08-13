#!/usr/bin/env python3
"""
fusion/03_fusion_eval.py

Applies the fusion decision rule (binary RF mismatch + three-band RF
confidence + HMAC verdict + freshness verdict -> accept / reject / flag),
runs the four canonical attack scenarios from the handoff spec, and reports
the headline metric: what the combined system catches that a higher-layer
-only (crypto-only) system misses.

------------------------------------------------------------------------
DECISION RULE (this is the one genuinely interpretive design choice in
this script -- confirm it reads right before trusting the numbers):

    1. hmac_pass is False        -> REJECT
    2. freshness_pass is False   -> REJECT
    3. otherwise (both pass), by RF confidence band:
         - "high"   -> ACCEPT if RF prediction matches the claimed
                        satellite, else REJECT
         - "middle" -> FLAG for inspection, regardless of match
         - "low"    -> FLAG for inspection, regardless of match

Rule 3's low/middle collapse (both -> flag, neither -> accept-by-default)
follows the handoff's own worked example directly: "Low RF confidence
(uncertain even if all pass) -> flag for inspection" is given as the
scenario's expected outcome, not "accept". The alternative reading -- treat
low confidence as the RF layer abstaining, and accept on higher-layer-pass
alone -- is defensible too, but isn't what the source table shows, so this
script does not use it. Flag if you want the alternative behaviour instead.
------------------------------------------------------------------------

Input:
    fusion/evidence.csv                (from 01_evidence_adapter.py)
    fusion/higher_layer_verdicts.csv   (from 02_higher_layer_sim.py)

Output:
    fusion/decision_table.csv          the four canonical scenarios, computed
    fusion/population_results.csv      full constructed-attack population
    fusion/headline_summary.txt        headline numbers, plain text
    fusion/headline_figure.png         combined vs higher-layer-only catch rates
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared import (
    COLUMN_MAP,
    SATELLITES,
    confidence_band,
    CONF_BAND_LOW_MAX,
    CONF_BAND_HIGH_MIN,
)

ID = COLUMN_MAP["id"]
TRUE = COLUMN_MAP["true_sat"]
CLAIMED = COLUMN_MAP["claimed_sat"]
PRED = COLUMN_MAP["predicted_sat"]
CONF = COLUMN_MAP["confidence"]


# --------------------------------------------------------------------------
# The decision rule
# --------------------------------------------------------------------------

def decide(match: bool, band: str, hmac_pass: bool, freshness_pass: bool) -> str:
    if not hmac_pass:
        return "reject"
    if not freshness_pass:
        return "reject"
    if band == "high":
        return "accept" if match else "reject"
    return "flag"  # band in {"low", "middle"}


def caught_by(hmac_pass, freshness_pass, band, match, decision) -> str:
    if decision == "accept":
        return "n/a (accepted)"
    if not hmac_pass:
        return "HMAC"
    if not freshness_pass:
        return "freshness"
    if decision == "reject":
        return "RF mismatch"
    return "confidence gate"


# --------------------------------------------------------------------------
# Band edges. Default to CONF_BAND_LOW_MAX / CONF_BAND_HIGH_MIN from
# _shared.py (tertiles, chosen because rf_confidence has no measurable
# relationship to correctness in this data -- AUC 0.506, p=0.77 -- so there
# is no discriminative threshold to optimise for; see _shared.py for the
# full justification). Recomputes tertiles from whatever evidence.csv is
# passed in if edges aren't supplied, so this still works on other subsets.
# --------------------------------------------------------------------------

def resolve_band_edges(evidence: pd.DataFrame, low_max=None, high_min=None):
    if low_max is not None and high_min is not None:
        return low_max, high_min
    q1, q2 = evidence[CONF].quantile([1 / 3, 2 / 3])
    print(
        f"No band edges supplied -- computing observed-confidence tertiles "
        f"for this file: low<= {q1:.4f}, high>= {q2:.4f}."
    )
    return q1, q2


# --------------------------------------------------------------------------
# Illustrative scenarios -- one concrete case per row of the handoff's table
# --------------------------------------------------------------------------

def build_illustrative_table(merged: pd.DataFrame, low_max, high_min) -> pd.DataFrame:
    rows = []

    # Pick a genuine, correctly-classified, HIGH-confidence message to
    # illustrate the two wrong-hardware scenarios (spoofed identity, stolen
    # key). Both assume the attacker holds valid key material -- that's the
    # premise the demo exists to address -- so hmac_pass/freshness_pass are
    # True by construction; only the RF layer can catch these.
    correct_hi = merged[
        (merged[PRED] == merged[TRUE])
        & (merged[CONF] >= high_min)
        & merged["hmac_pass"] & merged["freshness_pass"]
    ]
    if len(correct_hi) == 0:
        print("[warn] no high-confidence correctly-classified genuine message "
              "found for the wrong-hardware illustration; loosening to any "
              "confidence.")
        correct_hi = merged[(merged[PRED] == merged[TRUE])
                             & merged["hmac_pass"] & merged["freshness_pass"]]
    base = correct_hi.iloc[0]
    false_claim = next(s for s in SATELLITES if s != base[TRUE])

    for label in ("Spoofed identity", "Stolen / cloned key"):
        match = bool(base[PRED] == false_claim)
        band = confidence_band(base[CONF], low_max, high_min)
        d = decide(match, band, True, True)
        rows.append({
            "scenario": label,
            "global_index": base[ID],
            "true_sat": base[TRUE],
            "claimed_sat": false_claim,
            "rf_predicted_sat": base[PRED],
            "rf_confidence": round(float(base[CONF]), 4),
            "confidence_band": band,
            "rf_match": match,
            "hmac_pass": True,
            "freshness_pass": True,
            "decision": d,
            "caught_by": caught_by(True, True, band, match, d),
        })

    # Replay: a genuine message's own (sat, nonce) pair, verified a second
    # time. HMAC still validates (same content); freshness fails because the
    # nonce has already been consumed.
    genuine = merged[merged["hmac_pass"] & merged["freshness_pass"]].iloc[0]
    band = confidence_band(genuine[CONF], low_max, high_min)
    match = bool(genuine[PRED] == genuine[CLAIMED])
    d = decide(match, band, True, False)
    rows.append({
        "scenario": "Replay",
        "global_index": genuine[ID],
        "true_sat": genuine[TRUE],
        "claimed_sat": genuine[CLAIMED],
        "rf_predicted_sat": genuine[PRED],
        "rf_confidence": round(float(genuine[CONF]), 4),
        "confidence_band": band,
        "rf_match": match,
        "hmac_pass": True,
        "freshness_pass": False,
        "decision": d,
        "caught_by": caught_by(True, False, band, match, d),
    })

    # Low RF confidence, everything else clean.
    low_conf_rows = merged[(merged[CONF] <= low_max)
                            & merged["hmac_pass"] & merged["freshness_pass"]]
    if len(low_conf_rows) == 0:
        low_conf_rows = merged[merged["hmac_pass"] & merged["freshness_pass"]]
    lc = low_conf_rows.iloc[0]
    band = confidence_band(lc[CONF], low_max, high_min)
    match = bool(lc[PRED] == lc[CLAIMED])
    d = decide(match, band, True, True)
    rows.append({
        "scenario": "Low RF confidence",
        "global_index": lc[ID],
        "true_sat": lc[TRUE],
        "claimed_sat": lc[CLAIMED],
        "rf_predicted_sat": lc[PRED],
        "rf_confidence": round(float(lc[CONF]), 4),
        "confidence_band": band,
        "rf_match": match,
        "hmac_pass": True,
        "freshness_pass": True,
        "decision": d,
        "caught_by": caught_by(True, True, band, match, d),
    })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Population-level headline metric
# --------------------------------------------------------------------------

def build_population_results(merged: pd.DataFrame, low_max, high_min) -> pd.DataFrame:
    rows = []

    # (a) Genuine traffic, as-is: cost of the fusion layer to legitimate
    # messages. claimed == true for every row here, by construction of
    # evidence.csv (there is no live attacker in captured data).
    for _, r in merged.iterrows():
        band = confidence_band(r[CONF], low_max, high_min)
        match = bool(r[PRED] == r[CLAIMED])
        d = decide(match, band, bool(r["hmac_pass"]), bool(r["freshness_pass"]))
        rows.append({
            "case_type": "genuine",
            "global_index": r[ID], "true_sat": r[TRUE], "claimed_sat": r[CLAIMED],
            "rf_match": match, "confidence_band": band,
            "hmac_pass": r["hmac_pass"], "freshness_pass": r["freshness_pass"],
            "decision": d,
            "combined_caught": d == "reject",
            "higher_layer_only_caught": (not r["hmac_pass"]) or (not r["freshness_pass"]),
        })

    # (b) Wrong-hardware: every genuine message, re-asserted under every
    # OTHER satellite's identity. HMAC/freshness pass by construction
    # (valid key assumed -- the premise this scenario tests). RF prediction
    # is untouched, since the physical signal is unchanged.
    for _, r in merged.iterrows():
        for false_claim in SATELLITES:
            if false_claim == r[TRUE]:
                continue
            band = confidence_band(r[CONF], low_max, high_min)
            match = bool(r[PRED] == false_claim)
            d = decide(match, band, True, True)
            rows.append({
                "case_type": "wrong_hardware",
                "global_index": r[ID], "true_sat": r[TRUE], "claimed_sat": false_claim,
                "rf_match": match, "confidence_band": band,
                "hmac_pass": True, "freshness_pass": True,
                "decision": d,
                "combined_caught": d == "reject",
                "higher_layer_only_caught": False,  # crypto passes by construction
            })

    # (c) Replay: every genuine message's nonce, reused. HMAC still passes;
    # freshness fails deterministically (a consumed nonce is always caught).
    for _, r in merged.iterrows():
        band = confidence_band(r[CONF], low_max, high_min)
        match = bool(r[PRED] == r[CLAIMED])
        d = decide(match, band, True, False)
        rows.append({
            "case_type": "replay",
            "global_index": r[ID], "true_sat": r[TRUE], "claimed_sat": r[CLAIMED],
            "rf_match": match, "confidence_band": band,
            "hmac_pass": True, "freshness_pass": False,
            "decision": d,
            "combined_caught": d == "reject",
            "higher_layer_only_caught": True,  # freshness alone already catches replay
        })

    return pd.DataFrame(rows)


def summarise(pop: pd.DataFrame) -> str:
    lines = []
    for case_type in ["genuine", "wrong_hardware", "replay"]:
        sub = pop[pop["case_type"] == case_type]
        n = len(sub)
        if n == 0:
            continue
        dist = sub["decision"].value_counts(normalize=True).mul(100).round(1)
        lines.append(f"\n{case_type}  (n={n})")
        for k in ["accept", "flag", "reject"]:
            lines.append(f"  {k:7s}: {dist.get(k, 0.0):5.1f}%")
        if case_type != "genuine":
            combined = sub["combined_caught"].mean() * 100
            hlonly = sub["higher_layer_only_caught"].mean() * 100
            lines.append(f"  caught by combined system:      {combined:5.1f}%")
            lines.append(f"  caught by higher-layer-only:     {hlonly:5.1f}%")
    return "\n".join(lines)


def make_figure(pop: pd.DataFrame, path: str):
    cats, combined_rates, hl_rates = [], [], []
    for case_type, label in [("wrong_hardware", "Wrong-hardware\n(spoof / stolen key)"),
                              ("replay", "Replay")]:
        sub = pop[pop["case_type"] == case_type]
        if len(sub) == 0:
            continue
        cats.append(label)
        combined_rates.append(sub["combined_caught"].mean() * 100)
        hl_rates.append(sub["higher_layer_only_caught"].mean() * 100)

    x = range(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([i - width / 2 for i in x], hl_rates, width, label="Higher-layer-only (HMAC+freshness)")
    ax.bar([i + width / 2 for i in x], combined_rates, width, label="Combined (+ RF layer)")
    ax.set_ylabel("Attack catch rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_title("What the RF layer adds: attacks caught\nby combined vs. higher-layer-only security")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8)
    for i, (h, c) in enumerate(zip(hl_rates, combined_rates)):
        ax.text(i - width / 2, h + 2, f"{h:.0f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, c + 2, f"{c:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_note(evidence: pd.DataFrame) -> str:
    """Does rf_confidence actually distinguish correct RF predictions from
    incorrect ones? This is the check that justifies (or would rule out)
    deriving band edges from confidence at all."""
    from scipy.stats import mannwhitneyu
    correct = evidence.loc[evidence[PRED] == evidence[TRUE], CONF].values
    incorrect = evidence.loc[evidence[PRED] != evidence[TRUE], CONF].values
    u_stat, p_value = mannwhitneyu(correct, incorrect, alternative="two-sided")
    auc = u_stat / (len(correct) * len(incorrect))
    return (
        f"Confidence calibration check: does rf_confidence predict whether the\n"
        f"RF classifier was correct? AUC = {auc:.3f} (0.5 = no signal), "
        f"Mann-Whitney p = {p_value:.3f}.\n"
        f"{'Not distinguishable from chance -- band edges below are a design ' if p_value > 0.05 else 'Signal detected -- '}"
        f"{'choice for a roughly even split, not an optimised threshold.' if p_value > 0.05 else ''}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="fusion/evidence.csv")
    ap.add_argument("--verdicts", default="fusion/higher_layer_verdicts.csv")
    ap.add_argument("--low-max", type=float, default=CONF_BAND_LOW_MAX)
    ap.add_argument("--high-min", type=float, default=CONF_BAND_HIGH_MIN)
    ap.add_argument("--outdir", default="fusion")
    args = ap.parse_args()

    evidence = pd.read_csv(args.evidence)
    verdicts = pd.read_csv(args.verdicts)
    merged = evidence.merge(verdicts, on=ID, how="inner")
    if len(merged) != len(evidence):
        print(f"[warn] {len(evidence) - len(merged)} evidence rows had no matching "
              f"verdict row -- check the join key.")

    low_max, high_min = resolve_band_edges(evidence, args.low_max, args.high_min)

    illustrative = build_illustrative_table(merged, low_max, high_min)
    illustrative.to_csv(f"{args.outdir}/decision_table.csv", index=False)

    population = build_population_results(merged, low_max, high_min)
    population.to_csv(f"{args.outdir}/population_results.csv", index=False)

    summary = summarise(population)
    calib = calibration_note(evidence)
    with open(f"{args.outdir}/headline_summary.txt", "w") as f:
        f.write(calib + "\n\n")
        f.write(f"Band edges used: low<={low_max:.4f}, high>={high_min:.4f}\n")
        f.write(summary + "\n")

    make_figure(population, f"{args.outdir}/headline_figure.png")

    print("\n=== Confidence calibration (fusion/headline_summary.txt) ===")
    print(calib)
    print("\n=== Illustrative scenarios (fusion/decision_table.csv) ===")
    print(illustrative[["scenario", "rf_match", "confidence_band", "hmac_pass",
                         "freshness_pass", "decision", "caught_by"]].to_string(index=False))
    print("\n=== Population-level headline (fusion/population_results.csv) ===")
    print(summary)
    print(f"\nFigure written to {args.outdir}/headline_figure.png")


if __name__ == "__main__":
    main()
