#!/usr/bin/env python3
"""
32_thesis_tables.py — Reproducible thesis comparison tables
=============================================================
Reads validated experimental result CSVs from outputs/tables/ and
generates two summary tables. It does not train models, rerun
experiments, simulate authentication, or contain manually copied
performance numbers — every value is read from an existing result file.

Outputs:
  outputs/tables/model_comparison.csv
  outputs/tables/feature_table.csv

Threat-model and future-work documents are dissertation narrative and
are no longer generated here. The canonical multi-layer authentication
implementation lives under fusion/.
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)

def load_csv(name, required_columns=frozenset()):
    """Read a CSV from outputs/tables/ and validate it has the expected columns."""
    path = OUT_TABLES / name
    if not path.exists():
        fail(
            f"required source file is missing: {path}\n"
            f"  Run the pipeline script that produces '{name}' before "
            f"running 32_thesis_tables.py."
        )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path} exists but contains no data rows.")
    missing = set(required_columns) - set(rows[0].keys())
    if missing:
        fail(
            f"{path} is missing expected column(s): {sorted(missing)}\n"
            f"  Columns found: {sorted(rows[0].keys())}\n"
            f"  The upstream script's output schema may have changed — "
            f"update 32_thesis_tables.py to match it. Do not substitute "
            f"default values for a schema mismatch."
        )
    return rows

# TABLE 1: MODEL COMPARISON
#
# Long ("tidy") format: one row per (model, evaluation_protocol, metric).
# This is deliberate — the source CSVs report different metrics under
# different protocols, and forcing them into a single wide table would
# mean padding most cells with N/A or inventing values that were never
# computed. Every row here traces to exactly one cell in one source CSV
# (see source_csv), and a metric that isn't available for a given
# model/protocol is simply absent rather than filled in.

comparison_rows = []

def emit(model, protocol, metric, value, source_csv):
    if value is None or str(value).strip() == "":
        return  # metric unavailable for this row — omit, do not invent
    comparison_rows.append(
        {
            "model": model,
            "evaluation_protocol": protocol,
            "metric": metric,
            "value": value,
            "source_csv": source_csv,
        }
    )

# Stratified 80/20 split + 5-fold StratifiedKFold CV on the train split
# (see scripts/05_train_classifiers.py). This supersedes model_results.csv,
# an older output of the now-archived archive/04_train_classifiers.py with
# the same test-set accuracy/macro-F1 for every shared model but no CV or
# timing columns — model_results.csv is intentionally not read here.
cc_rows = load_csv(
    "classifier_comparison.csv",
    {
        "model", "scaled", "cv_accuracy_mean", "cv_accuracy_std",
        "test_accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "test_macro_f1", "train_eval_seconds",
    },
)
CC_PROTOCOL = "Stratified 80/20 split + 5-fold CV (message-level, classical/MLP baselines)"
for r in cc_rows:
    model = r["model"]
    emit(model, CC_PROTOCOL, "accuracy", r["test_accuracy"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "accuracy_ci_low", r["accuracy_ci_low"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "accuracy_ci_high", r["accuracy_ci_high"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "macro_f1", r["test_macro_f1"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "cv_accuracy_mean", r["cv_accuracy_mean"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "cv_accuracy_std", r["cv_accuracy_std"], "classifier_comparison.csv")
    emit(model, CC_PROTOCOL, "train_eval_seconds", r["train_eval_seconds"], "classifier_comparison.csv")
    emit(
        model, CC_PROTOCOL, "feature_scaling",
        "scaled" if r["scaled"].strip().lower() in ("true", "1", "yes") else "unscaled",
        "classifier_comparison.csv",
    )

# Pass/session-aware grouped cross-validation (see
# scripts/06_pass_aware_evaluation.py) — folds are grouped by satellite
# pass rather than shuffled, so the split tests generalisation to a new
# pass rather than to held-out messages from a pass already seen in training.
pa_rows = load_csv(
    "pass_aware_results.csv",
    {
        "model", "scaled", "mean_accuracy", "std_accuracy",
        "mean_macro_f1", "std_macro_f1", "min_fold_accuracy", "max_fold_accuracy",
    },
)
PA_PROTOCOL = "Pass-aware grouped cross-validation (session/pass-level, classical/MLP baselines)"
for r in pa_rows:
    model = r["model"]
    emit(model, PA_PROTOCOL, "accuracy", r["mean_accuracy"], "pass_aware_results.csv")
    emit(model, PA_PROTOCOL, "accuracy_std", r["std_accuracy"], "pass_aware_results.csv")
    emit(model, PA_PROTOCOL, "macro_f1", r["mean_macro_f1"], "pass_aware_results.csv")
    emit(model, PA_PROTOCOL, "macro_f1_std", r["std_macro_f1"], "pass_aware_results.csv")
    emit(model, PA_PROTOCOL, "min_fold_accuracy", r["min_fold_accuracy"], "pass_aware_results.csv")
    emit(model, PA_PROTOCOL, "max_fold_accuracy", r["max_fold_accuracy"], "pass_aware_results.csv")
    emit(
        model, PA_PROTOCOL, "feature_scaling",
        "scaled" if r["scaled"].strip().lower() in ("true", "1", "yes") else "unscaled",
        "pass_aware_results.csv",
    )

# Feature-based PyTorch MLP (scripts/27_mlp_features.py), trained on the
# same 28-feature set as script 05, but a distinct implementation from the
# scikit-learn "Neural Net (MLP)" rows above — kept as separate model labels
# so the two are never conflated.
mlp_rows = load_csv("mlp_results.csv", {"model", "evaluation", "accuracy", "macro_f1"})
for r in mlp_rows:
    model = f"{r['model']} (feature-based, PyTorch)"
    protocol = r["evaluation"]
    emit(model, protocol, "accuracy", r["accuracy"], "mlp_results.csv")
    emit(model, protocol, "macro_f1", r["macro_f1"], "mlp_results.csv")

# 1D-CNN on raw IQ bursts, 3-fold cross-session GroupKFold
# (scripts/22_cnn_raw_iq.py). Per-fold rows remain in cnn_results.csv for
# traceability, while this comparison table uses only the aggregate mean.
cnn_rows = load_csv(
    "cnn_results.csv", {"fold", "test_session", "n_train", "n_test", "accuracy", "macro_f1"}
)
CNN_MODEL = "1D-CNN (raw IQ)"
cnn_mean_rows = [r for r in cnn_rows if r["fold"] == "mean"]
if not cnn_mean_rows:
    fail("cnn_results.csv does not contain the required fold == 'mean' row")
for r in cnn_mean_rows:
    protocol = "Cross-session GroupKFold (raw IQ, 3-fold mean)"
    emit(CNN_MODEL, protocol, "accuracy", r["accuracy"], "cnn_results.csv")
    emit(CNN_MODEL, protocol, "macro_f1", r["macro_f1"], "cnn_results.csv")
    emit(CNN_MODEL, protocol, "n_train", r["n_train"], "cnn_results.csv")
    emit(CNN_MODEL, protocol, "n_test", r["n_test"], "cnn_results.csv")

# Per-satellite FRR/FAR/EER (scripts/20_authentication_metrics.py), reframing
# classification as an authentication decision. model_accuracy/model_EER are
# validated as constant across the per-satellite rows for a given feature set,
# then each is emitted once; n_genuine is summed across satellites into
# n_genuine_total.
auth_rows = load_csv(
    "authentication_metrics.csv",
    {"model", "satellite", "n_genuine", "FRR", "FAR", "model_accuracy", "model_EER"},
)
AUTH_PROTOCOL = "Authentication / EER evaluation (argmax decision rule + threshold sweep)"
auth_by_model = {}
for r in auth_rows:
    agg = auth_by_model.setdefault(
        r["model"], {"n_genuine": 0, "model_accuracy": set(), "model_EER": set()}
    )
    agg["n_genuine"] += int(r["n_genuine"])
    if r["model_accuracy"].strip():
        agg["model_accuracy"].add(r["model_accuracy"])
    if r["model_EER"].strip():
        agg["model_EER"].add(r["model_EER"])
for model, agg in auth_by_model.items():
    for metric in ("model_accuracy", "model_EER"):
        values = agg[metric]
        if len(values) > 1:
            fail(f"authentication_metrics.csv has conflicting {metric} values for model '{model}': {sorted(values)}")
        if not values:
            fail(f"authentication_metrics.csv has no non-empty {metric} value for model '{model}'")
        agg[metric] = next(iter(values))
    emit(model, AUTH_PROTOCOL, "accuracy", agg["model_accuracy"], "authentication_metrics.csv")
    emit(model, AUTH_PROTOCOL, "EER", agg["model_EER"], "authentication_metrics.csv")
    emit(model, AUTH_PROTOCOL, "n_genuine_total", str(agg["n_genuine"]), "authentication_metrics.csv")

with open(OUT_TABLES / "model_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "evaluation_protocol", "metric", "value", "source_csv"])
    w.writeheader()
    w.writerows(comparison_rows)
print(f"Saved: {OUT_TABLES / 'model_comparison.csv'}  ({len(comparison_rows)} rows)")

# TABLE 2: FEATURE TABLE
#
# Definitions/groupings below are configuration facts taken directly from
# the extraction scripts' own code and comments (04_extract_features.py,
# 15_extract_features_v2.py, 17_extract_features_v3.py) — not measured
# results. Every feature name is validated against the real CSV header of
# the file that script produces, so this table cannot silently drift out
# of sync with the extraction code. R^2-vs-level values and leakage
# verdicts are joined in from the existing analysis at run time rather
# than copied as literals.

FEATURES_V1 = [
    ("mean_I", "Time-domain amplitude and shape", "mean(I)",
     "May reflect a small DC offset introduced by the modulator on the I arm; could also be receiver-side estimation noise."),
    ("mean_Q", "Time-domain amplitude and shape", "mean(Q)",
     "May reflect a small DC offset introduced by the modulator on the Q arm; could also be receiver-side estimation noise."),
    ("var_I", "Time-domain amplitude and shape", "var(I)",
     "Can be influenced by received signal power (path loss, fading); not on its own a hardware-specific measure."),
    ("var_Q", "Time-domain amplitude and shape", "var(Q)",
     "Can be influenced by received signal power (path loss, fading); not on its own a hardware-specific measure."),
    ("std_I", "Time-domain amplitude and shape", "sqrt(var(I))",
     "Closely tracks received signal amplitude; may say more about link conditions than about the transmitter (see Channel_Dominance_R2_vs_Level)."),
    ("std_Q", "Time-domain amplitude and shape", "sqrt(var(Q))",
     "Closely tracks received signal amplitude; may say more about link conditions than about the transmitter (see Channel_Dominance_R2_vs_Level)."),
    ("max_I", "Time-domain amplitude and shape", "max(I)",
     "Peak-amplitude measure; can be influenced by received power and by front-end clipping/compression."),
    ("max_Q", "Time-domain amplitude and shape", "max(Q)",
     "Peak-amplitude measure; can be influenced by received power and by front-end clipping/compression."),
    ("min_I", "Time-domain amplitude and shape", "min(I)",
     "Negative peak-amplitude measure; can be influenced by received power and by front-end clipping/compression."),
    ("min_Q", "Time-domain amplitude and shape", "min(Q)",
     "Negative peak-amplitude measure; can be influenced by received power and by front-end clipping/compression."),
    ("median_I", "Time-domain amplitude and shape", "median(I)",
     "Robust centre estimate; may reflect the same small DC-offset effects as mean_I but is less sensitive to outliers."),
    ("median_Q", "Time-domain amplitude and shape", "median(Q)",
     "Robust centre estimate; may reflect the same small DC-offset effects as mean_Q but is less sensitive to outliers."),
    ("iqr_I", "Time-domain amplitude and shape", "P75(I) - P25(I)",
     "Robust spread measure; empirically tracks received amplitude closely (see Channel_Dominance_R2_vs_Level), so may primarily reflect link conditions."),
    ("iqr_Q", "Time-domain amplitude and shape", "P75(Q) - P25(Q)",
     "Robust spread measure; empirically tracks received amplitude closely (see Channel_Dominance_R2_vs_Level), so may primarily reflect link conditions."),
    ("skew_I", "Distribution shape (scale-independent)", "skewness(I)",
     "Dimensionless asymmetry measure; potentially associated with transmit-chain amplitude-distribution effects, not isolated from other sources by this script."),
    ("skew_Q", "Distribution shape (scale-independent)", "skewness(Q)",
     "Dimensionless asymmetry measure; potentially associated with transmit-chain amplitude-distribution effects, not isolated from other sources by this script."),
    ("kurt_I", "Distribution shape (scale-independent)", "kurtosis(I)",
     "Tailedness of the amplitude distribution; can be influenced by clipping/compression in the transmit chain, but may also reflect noise or interference."),
    ("kurt_Q", "Distribution shape (scale-independent)", "kurtosis(Q)",
     "Tailedness of the amplitude distribution; can be influenced by clipping/compression in the transmit chain, but may also reflect noise or interference."),
    ("signal_power", "Power and I/Q relationship", "mean(I^2 + Q^2)",
     "Direct measure of received power; typically dominated by path loss, antenna gain and fading rather than transmitter identity."),
    ("iq_ratio", "Power and I/Q relationship", "std(I) / std(Q)",
     "Crude proxy for gain imbalance between the I and Q channels; potentially associated with analogue front-end characteristics, not validated against a ground-truth imbalance measurement."),
    ("iq_correlation", "Power and I/Q relationship", "corr(I, Q)",
     "Near zero for ideal QPSK; a non-zero value could indicate quadrature phase error, but this script does not establish whether it originates at the transmitter or receiver."),
    ("papr", "Power and I/Q relationship", "max(I^2+Q^2) / mean(I^2+Q^2)",
     "Peak-to-average power ratio; can be influenced by amplifier compression, but also by multipath, which can create additional peaks."),
    ("zero_crossing_rate", "Power and I/Q relationship", "rate of sign changes in I",
     "Coarse frequency-domain proxy; potentially influenced by carrier offset, but is a rough indicator rather than a calibrated frequency measurement."),
    ("fft_mean_magnitude", "Frequency-domain", "mean(|FFT(I + jQ)|)",
     "Overall spectral energy; scales with received power in the same way as the time-domain amplitude features."),
    ("peak_frequency", "Frequency-domain", "argmax_f |FFT(I + jQ)|",
     "May reflect carrier frequency offset from transmitter oscillator error, but this is confounded with Doppler shift, which is geometry-dependent and can be much larger."),
    ("spectral_centroid", "Frequency-domain", "sum(f * |X(f)|) / sum(|X(f)|)",
     "Amplitude-weighted mean frequency; subject to the same carrier-offset/Doppler confound as peak_frequency."),
    ("bandwidth", "Frequency-domain", "spectral spread about the centroid",
     "Primarily a property of the modulation and pulse shaping rather than of the individual transmitter."),
    ("occupied_bandwidth", "Frequency-domain", "width containing 90% of spectral energy",
     "Primarily a structural/protocol property rather than a hardware-specific measure."),
]

FEATURES_V2 = [
    ("cfo_hz", "Carrier frequency offset & oscillator (4th-power method)", "mean(inst. freq. of z^4) / 4",
        "Mean carrier frequency offset. This can contain contributions from both transmitter oscillator error and "
        "satellite-receiver Doppler shift. Consequently, it cannot be attributed to transmitter hardware alone "
        "without reliable Doppler compensation."),
    ("cfo_std_hz", "Carrier frequency offset & oscillator (4th-power method)", "std(inst. freq. of z^4) / 4",
     "Short-term spread of the instantaneous-frequency estimate; may reflect oscillator instability/phase noise, subject to the same CFO/Doppler confound as cfo_hz."),
    ("cfo_iqr_hz", "Carrier frequency offset & oscillator (4th-power method)", "P75 - P25 of inst. freq.",
     "Robust version of cfo_std_hz, less sensitive to residual symbol-transition spikes."),
    ("cfo_drift_hz_per_ms", "Carrier frequency offset & oscillator (4th-power method)", "linear slope of inst. freq. vs time",
     "Linear frequency drift across the burst; potentially associated with oscillator warm-up behaviour, not isolated from any Doppler rate-of-change."),
    ("cfo_skew", "Carrier frequency offset & oscillator (4th-power method)", "skewness(inst. freq.)",
     "Shape (asymmetry) of the instantaneous-frequency distribution within a burst."),
    ("cfo_kurt", "Carrier frequency offset & oscillator (4th-power method)", "kurtosis(inst. freq.)",
     "Shape (tailedness) of the instantaneous-frequency distribution; may help distinguish smooth drift from impulsive phase disturbance."),
    ("iq_gain_imbalance_db", "I/Q imbalance", "20*log10(std(I)/std(Q))",
     "Gain imbalance between the I and Q channels; potentially associated with analogue quadrature-modulator characteristics of the transmitter."),
    ("iq_phase_error_deg", "I/Q imbalance", "arcsin(corr(I, Q))",
     "Estimated quadrature phase error, recovered from I/Q correlation; potentially associated with transmitter modulator characteristics."),
    ("dc_offset_i", "I/Q imbalance", "mean(I) after unit-power normalisation",
     "Residual DC leakage on the I arm, expressed relative to signal amplitude; potentially associated with modulator/mixer characteristics."),
    ("dc_offset_q", "I/Q imbalance", "mean(Q) after unit-power normalisation",
     "Residual DC leakage on the Q arm, expressed relative to signal amplitude; potentially associated with modulator/mixer characteristics."),
    ("dc_offset_mag", "I/Q imbalance", "hypot(dc_offset_i, dc_offset_q)",
     "Combined magnitude of I/Q DC leakage; potentially associated with modulator/mixer characteristics."),
    ("envelope_cv", "Envelope behaviour", "std(|z|) / mean(|z|)",
     "Envelope ripple relative to its own level; amplitude-invariant by construction (see extraction script rationale)."),
    ("envelope_slope", "Envelope behaviour", "linear trend of |z| over the burst, normalised",
     "Potentially associated with amplifier droop or transmit power-control action during the burst."),
    ("papr_db", "Envelope behaviour", "10*log10(max power / mean power)",
     "Peak-to-average power ratio, amplitude-invariant by construction — distinct from the unnormalised v1 papr feature."),
    ("envelope_kurt", "Envelope behaviour", "kurtosis(|z|)",
     "Tailedness of the envelope distribution; can be influenced by clipping/compression."),
    ("constellation_spread_deg", "Constellation quality", "circular std of angle(z^4)",
     "Angular spread of the residual (post-4th-power) constellation; may reflect how tightly the transmitter holds its constellation, independent of any fixed rotation."),
    ("evm_proxy", "Constellation quality", "std(|z|) / mean(|z|)",
     "Magnitude dispersion about the mean radius; a proxy for error vector magnitude that does not require symbol decisions."),
    ("phase_jitter", "Constellation quality", "std(2nd difference of unwrapped phase(z^4))",
     "Short-term phase instability after removing constant offset and linear drift."),
    ("seg1_power_ratio", "Power trajectory across sub-windows", "mean power of segment 1 / mean power of burst",
     "Power of one quarter of the burst relative to the message mean; captures coarse within-burst power trajectory (amplitude-invariant by construction)."),
    ("seg2_power_ratio", "Power trajectory across sub-windows", "mean power of segment 2 / mean power of burst",
     "Power of one quarter of the burst relative to the message mean; captures coarse within-burst power trajectory (amplitude-invariant by construction)."),
    ("seg3_power_ratio", "Power trajectory across sub-windows", "mean power of segment 3 / mean power of burst",
     "Power of one quarter of the burst relative to the message mean; captures coarse within-burst power trajectory (amplitude-invariant by construction)."),
    ("seg4_power_ratio", "Power trajectory across sub-windows", "mean power of segment 4 / mean power of burst",
     "Power of one quarter of the burst relative to the message mean; captures coarse within-burst power trajectory (amplitude-invariant by construction)."),
    ("skew_i_norm", "Normalised distribution shape", "skewness(I) after unit-power normalisation",
     "Distribution shape of the amplitude-normalised I samples; recomputed on normalised data to avoid conflation with the raw-power features in v1."),
    ("skew_q_norm", "Normalised distribution shape", "skewness(Q) after unit-power normalisation",
     "Distribution shape of the amplitude-normalised Q samples; recomputed on normalised data to avoid conflation with the raw-power features in v1."),
    ("kurt_i_norm", "Normalised distribution shape", "kurtosis(I) after unit-power normalisation",
     "Distribution shape of the amplitude-normalised I samples; recomputed on normalised data to avoid conflation with the raw-power features in v1."),
    ("kurt_q_norm", "Normalised distribution shape", "kurtosis(Q) after unit-power normalisation",
     "Distribution shape of the amplitude-normalised Q samples; recomputed on normalised data to avoid conflation with the raw-power features in v1."),
]

FEATURES_V3 = [
    ("ampm_correlation", "AM/PM conversion", "corr(residual phase, amplitude)",
     "Zero for an ideal amplifier; a non-zero value can be influenced by AM/PM conversion in the transmit power amplifier."),
    ("ampm_slope_deg", "AM/PM conversion", "slope of residual phase vs amplitude",
     "AM/PM conversion coefficient; per the extraction script's stated rationale, potentially associated with the individual amplifier's bias point and semiconductor characteristics."),
    ("ampm_q1_deg", "AM/PM conversion", "mean residual phase, lowest amplitude quartile",
     "A linear amplifier would show a similar value in every amplitude quartile; deviation can indicate nonlinearity."),
    ("ampm_q4_deg", "AM/PM conversion", "mean residual phase, highest amplitude quartile",
     "A linear amplifier would show a similar value in every amplitude quartile; deviation can indicate nonlinearity."),
    ("ampm_range_deg", "AM/PM conversion", "max(quartile phase) - min(quartile phase)",
     "Total phase excursion across amplitude quartiles; can be influenced by amplifier compression."),
    ("ampm_curvature", "AM/PM conversion", "2nd-order coefficient of phase-vs-amplitude fit",
     "Curvature of the phase-amplitude relationship, beyond the linear slope term."),
    ("ccdf_1db", "AM/AM compression (CCDF)", "P(instantaneous power > mean + 1 dB)",
     "A shortened tail relative to an undistorted signal can indicate amplifier compression near saturation."),
    ("ccdf_3db", "AM/AM compression (CCDF)", "P(instantaneous power > mean + 3 dB)",
     "A shortened tail relative to an undistorted signal can indicate amplifier compression near saturation."),
    ("ccdf_6db", "AM/AM compression (CCDF)", "P(instantaneous power > mean + 6 dB)",
     "A shortened tail relative to an undistorted signal can indicate amplifier compression near saturation."),
    ("ccdf_9db", "AM/AM compression (CCDF)", "P(instantaneous power > mean + 9 dB)",
     "A shortened tail relative to an undistorted signal can indicate amplifier compression near saturation."),
    ("papr_deficit_db", "AM/AM compression (CCDF)", "measured PAPR (dB) - 5 dB reference",
     "PAPR relative to the ~5 dB expected for an undistorted pulse-shaped QPSK signal; a markedly lower value can indicate compression."),
    ("acpr_lower_db", "Spectral regrowth (ACPR)", "power in lower adjacent band / power in occupied band",
     "Intermodulation products from amplifier nonlinearity fall outside the occupied bandwidth, so this can be influenced by amplifier linearity."),
    ("acpr_upper_db", "Spectral regrowth (ACPR)", "power in upper adjacent band / power in occupied band",
     "Intermodulation products from amplifier nonlinearity fall outside the occupied bandwidth, so this can be influenced by amplifier linearity."),
    ("acpr_mean_db", "Spectral regrowth (ACPR)", "mean(acpr_lower_db, acpr_upper_db)",
     "Average adjacent-channel power ratio; can be influenced by amplifier linearity."),
    ("acpr_asymmetry_db", "Spectral regrowth (ACPR)", "acpr_upper_db - acpr_lower_db",
     "Per the extraction script's rationale, symmetric regrowth is expected from amplitude compression alone, while asymmetry may indicate amplitude/phase distortion interacting."),
    ("occupied_fraction", "Spectral regrowth (ACPR)", "occupied bandwidth / capture bandwidth",
     "A wider occupied band for the same protocol can imply greater spectral regrowth."),
    ("const_spread_q1_deg", "Constellation quality by amplitude", "circular std of angle(z^4), amplitude quartile 1",
     "Stratifying constellation tightness by amplitude can separate distortion that grows with drive level from distortion that does not."),
    ("const_spread_q2_deg", "Constellation quality by amplitude", "circular std of angle(z^4), amplitude quartile 2",
     "Stratifying constellation tightness by amplitude can separate distortion that grows with drive level from distortion that does not."),
    ("const_spread_q3_deg", "Constellation quality by amplitude", "circular std of angle(z^4), amplitude quartile 3",
     "Stratifying constellation tightness by amplitude can separate distortion that grows with drive level from distortion that does not."),
    ("const_spread_q4_deg", "Constellation quality by amplitude", "circular std of angle(z^4), amplitude quartile 4",
     "Stratifying constellation tightness by amplitude can separate distortion that grows with drive level from distortion that does not."),
]

FEATURE_SETS = [
    ("04_extract_features.py", "features.csv", FEATURES_V1,
     "Yes — this is the feature set used by the classifiers reported under the "
     "\"Stratified 80/20 split + 5-fold CV\" and \"Pass-aware\" protocols in model_comparison.csv."),
    ("15_extract_features_v2.py", "features_v2.csv", FEATURES_V2,
     "No — tested only in the separate v1/v2/v3 feature-set comparison "
     "(outputs/tables/feature_set_comparison.csv, outputs/tables/v1_v2_comparison.csv); "
     "not part of the feature set used by the classifiers reported in model_comparison.csv."),
    ("17_extract_features_v3.py", "features_v3.csv", FEATURES_V3,
     "No — tested only in the separate v1/v2/v3 feature-set comparison "
     "(outputs/tables/feature_set_comparison.csv); not part of the feature set used by the "
     "classifiers reported in model_comparison.csv."),
]

for source_script, csv_name, defs, _used in FEATURE_SETS:
    path = OUT_TABLES / csv_name
    if not path.exists():
        fail(f"required source file is missing: {path} (produced by scripts/{source_script})")
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    actual_features = set(header) - {"global_index", "satellite_id"}
    documented_features = {row[0] for row in defs}
    missing_from_table = actual_features - documented_features
    extra_in_table = documented_features - actual_features
    if missing_from_table:
        fail(
            f"{csv_name} (from {source_script}) has column(s) not documented in "
            f"32_thesis_tables.py: {sorted(missing_from_table)}\n"
            f"  Add a definition for each before regenerating feature_table.csv, so the "
            f"table cannot silently omit a feature the pipeline actually produces."
        )
    if extra_in_table:
        fail(
            f"32_thesis_tables.py documents feature(s) not present in {csv_name}: "
            f"{sorted(extra_in_table)}\n"
            f"  Remove them — the feature table must not claim a feature is used if it "
            f"is not actually produced by {source_script}."
        )

channel_dominance_rows = load_csv("channel_dominance.csv", {"feature", "r2_vs_level"})
r2_by_feature = {r["feature"]: r["r2_vs_level"] for r in channel_dominance_rows}

leakage_rows = load_csv("leakage_audit.csv", {"column", "accuracy", "chance", "verdict"})

def leakage_category(verdict):
    if "leak" in verdict.lower():
        return "A. Confirmed leakage / shortcut variable"
    return "B. Contextual/channel/receiver metadata (excluded; not demonstrated as leakage)"

METADATA_DESCRIPTIONS = {
    "global_index": (
        "Excluded metadata (dataset ordering)",
        "Sequential position of the message in the full recorded dataset.",
        "Encodes capture order rather than any transmitter characteristic.",
    ),
    "timestamp_global": (
        "Excluded metadata (dataset ordering)",
        "Capture timestamp of the message.",
        "Encodes when a message was recorded rather than who sent it.",
    ),
    "ra_lon": (
        "Excluded metadata (receiver-side / geometry)",
        "Receiver antenna longitude.",
        "Ground-station position; flagged by the leakage audit as above chance and worth further investigation, "
        "though far below the severity seen for global_index/timestamp_global.",
    ),
    "ra_lat": (
        "Excluded metadata (receiver-side / geometry)",
        "Receiver antenna latitude.",
        "Ground-station position; the leakage audit found no signal above chance for this column alone.",
    ),
    "ra_alt": (
        "Excluded metadata (receiver-side / geometry)",
        "Receiver antenna altitude.",
        "Ground-station position; the leakage audit found only a weak signal for this column alone.",
    ),
    "ra_cell": (
        "Excluded metadata (receiver-side)",
        "Receiver cell/beam identifier.",
        "Receiver-side identifier; the leakage audit found only a weak signal for this column alone.",
    ),
    "run_id": (
        "Excluded metadata (dataset/session)",
        "Recording run identifier.",
        "Dataset/session bookkeeping field; the leakage audit found no signal above chance for this column alone.",
    ),
    "confidence": (
        "Excluded metadata (receiver-side)",
        "Receiver's decode-confidence score for the message.",
        "Receiver-side quality metric; the leakage audit found no signal above chance for this column alone.",
    ),
    "direction": (
        "Excluded metadata (link geometry)",
        "Direction field recorded for the message.",
        "Link/geometry metadata; the leakage audit found no signal above chance for this column alone.",
    ),
    "level": (
        "Excluded metadata (channel)",
        "Receiver-reported signal level.",
        "Channel/link-budget metadata; the leakage audit found no signal above chance for this column alone.",
    ),
    "noise": (
        "Excluded metadata (channel)",
        "Receiver-reported noise level.",
        "Channel/link-budget metadata; the leakage audit found no signal above chance for this column alone.",
    ),
    "center_frequency": (
        "Excluded metadata (receiver-side)",
        "Receiver tuning centre frequency.",
        "Receiver-side configuration field; the leakage audit found no signal above chance for this column alone.",
    ),
}

feature_table_rows = []

for source_script, csv_name, defs, used_text in FEATURE_SETS:
    for name, group, formula, interpretation in defs:
        feature_table_rows.append(
            {
                "Feature": name,
                "Source": source_script,
                "Group": group,
                "Formula_Definition": formula,
                "Possible_Physical_Interpretation": interpretation,
                "Category": "C. RF-fingerprint candidate feature (waveform-derived)",
                "Used_By_Primary_Classifier": used_text,
                "Channel_Dominance_R2_vs_Level": r2_by_feature.get(name, "N/A"),
                "Leakage_Audit_Evidence": "N/A — not a metadata column",
            }
        )

documented_metadata = set(METADATA_DESCRIPTIONS)
audited_metadata = {r["column"] for r in leakage_rows}
missing_descriptions = audited_metadata - documented_metadata
if missing_descriptions:
    fail(
        f"leakage_audit.csv audits column(s) with no description in "
        f"32_thesis_tables.py: {sorted(missing_descriptions)}\n"
        f"  Add a METADATA_DESCRIPTIONS entry for each before regenerating feature_table.csv."
    )

for r in leakage_rows:
    column = r["column"]
    group, meaning, interpretation = METADATA_DESCRIPTIONS[column]
    evidence = (
        f"verdict='{r['verdict']}'; column-alone accuracy={float(r['accuracy']):.3f} "
        f"vs chance={float(r['chance']):.3f}"
    )
    feature_table_rows.append(
        {
            "Feature": column,
            "Source": "raw metadata (data/raw)",
            "Group": group,
            "Formula_Definition": meaning,
            "Possible_Physical_Interpretation": interpretation,
            "Category": leakage_category(r["verdict"]),
            "Used_By_Primary_Classifier": "No — excluded from the RF-fingerprint feature set",
            "Channel_Dominance_R2_vs_Level": "N/A — not part of the v1 waveform feature set",
            "Leakage_Audit_Evidence": evidence,
        }
    )

with open(OUT_TABLES / "feature_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(feature_table_rows[0].keys()))
    w.writeheader()
    w.writerows(feature_table_rows)
print(f"Saved: {OUT_TABLES / 'feature_table.csv'}  ({len(feature_table_rows)} rows)")

print("\nDone. 32_thesis_tables.py generates only model_comparison.csv and "
      "feature_table.csv — no reports, no threat model, no future work.")
