"""
09_statistical_rigour.py
========================

PURPOSE
-------
Addresses five methodological weaknesses in the analysis so far. Each
section corrects a specific overclaim or missing test.

  A. PASS-AWARE EVALUATION (GroupKFold)
     The random train/test split lets messages from the same satellite pass
     appear in both training and test data. Messages captured seconds apart
     share channel conditions, so a model can score above chance by
     recognising the pass rather than the transmitter. Passes are recovered
     from timestamp gaps and used as CV groups, so no pass is ever split.

  B. McNEMAR'S TEST
     Comparing two classifiers by their accuracy point estimates ignores
     that they were evaluated on the SAME test set, so their errors are
     paired. McNemar's test uses only the discordant cases -- messages one
     model got right and the other got wrong -- which is the correct test
     for paired classifier comparison.

  C. MAHALANOBIS SEPARABILITY
     The earlier separability figure used Euclidean distance between class
     means with an ad-hoc normalisation. Euclidean distance is misleading
     when features are correlated -- and features such as std_I, var_I and
     iqr_I are near-duplicates, so the same underlying quantity is counted
     several times. Mahalanobis distance accounts for the within-class
     covariance and is the standard measure.

  D. VARIANCE DECOMPOSITION OF `level`
     Previously it was shown that many features correlate with `level` at
     R^2 ~ 0.99, and this was described as channel dominance. That is close
     to circular: `level` is itself an amplitude estimate, so amplitude
     features must correlate with it. The substantive question is what
     drives `level`. This section attributes `level` variance to elevation
     angle, to beam (ra_cell), and to residual causes.

  E. WITHIN-BEAM CLASSIFICATION
     If beam assignment explains signal level, then holding beam constant
     is a stronger channel control than holding elevation constant.

USAGE
-----
    python scripts/09_statistical_rigour.py

OUTPUTS
-------
    outputs/tables/groupkfold_results.csv
    outputs/tables/mahalanobis_distances.csv
    outputs/tables/level_variance_decomposition.csv
    outputs/reports/statistical_rigour.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
N_BOOTSTRAP   = 2000
N_SPLITS      = 5

NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}
METADATA_COLUMNS = ["level", "noise", "ra_alt", "center_frequency"]

# A satellite is visible from a fixed ground station for roughly 10 minutes,
# and its orbital period is about 100 minutes. A gap of 20 minutes between
# consecutive messages from the same satellite therefore reliably indicates
# a new pass.
PASS_GAP_SECONDS = 20 * 60


# --- LOADING -------------------------------------------------------------
def load_metadata_column(column: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    return np.concatenate([np.load(f) for f in files]) if files else None


def load_all() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    features = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    key = next((c for c in ("global_index", "sample_id") if c in df.columns), None)
    if key is None:
        raise KeyError("No global_index / sample_id column.")
    rows = df[key].to_numpy(dtype=int)

    for col in METADATA_COLUMNS + ["ra_cell", "timestamp_global",
                                   "ra_lat", "ra_lon"]:
        full = load_metadata_column(col)
        if full is not None and rows.max() < len(full):
            df[f"meta_{col}"] = full[rows]

    return df, features


# --- ELEVATION (same corrected method as script 06) ----------------------
RECEIVER_LAT_DEG, RECEIVER_LON_DEG, RECEIVER_ALT_M = 51.7548, -1.2544, 60.0
WGS84_A, WGS84_F = 6378137.0, 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def add_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute elevation angle per message, so that section D can attribute
    `level` variance to geometry. Uses the same corrected treatment as
    script 06: ra_alt is GEOCENTRIC RADIUS, not height above the surface.

    Messages whose computed elevation is negative are set to NaN -- a
    satellite below the horizon cannot have delivered the message, so the
    position field must have been mis-decoded.
    """
    need = ("meta_ra_lat", "meta_ra_lon", "meta_ra_alt")
    if not all(c in df.columns for c in need):
        return df

    lat = df["meta_ra_lat"].to_numpy(float)
    lon = df["meta_ra_lon"].to_numpy(float)
    if np.nanmax(np.abs(lat)) <= 2.0:          # radians
        lat, lon = np.degrees(lat), np.degrees(lon)

    radial = df["meta_ra_alt"].to_numpy(float)
    med = float(np.nanmedian(radial))
    if 6_000 < med < 10_000:                   # geocentric radius, km
        radius_m = radial * 1000.0
    elif 6e6 < med < 1e7:                      # geocentric radius, m
        radius_m = radial
    elif 100 < med < 2_000:                    # altitude, km
        radius_m = radial * 1000.0 + 6_371_000.0
    else:
        radius_m = radial + 6_371_000.0

    lat0, lon0 = np.radians(RECEIVER_LAT_DEG), np.radians(RECEIVER_LON_DEG)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat0) ** 2)
    rx = ((N + RECEIVER_ALT_M) * np.cos(lat0) * np.cos(lon0),
          (N + RECEIVER_ALT_M) * np.cos(lat0) * np.sin(lon0),
          (N * (1.0 - WGS84_E2) + RECEIVER_ALT_M) * np.sin(lat0))

    la, lo = np.radians(lat), np.radians(lon)
    dx = radius_m * np.cos(la) * np.cos(lo) - rx[0]
    dy = radius_m * np.cos(la) * np.sin(lo) - rx[1]
    dz = radius_m * np.sin(la)                 - rx[2]

    east  = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = (-np.sin(lat0) * np.cos(lon0) * dx
             - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz)
    up    = ( np.cos(lat0) * np.cos(lon0) * dx
             + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz)

    elev = np.degrees(np.arctan2(up, np.sqrt(east ** 2 + north ** 2)))
    elev[elev < 0] = np.nan
    df["elevation_deg"] = elev
    print(f"  elevation computed: {np.isfinite(elev).sum():,} valid, "
          f"{np.isnan(elev).sum():,} below horizon (discarded)")
    return df


# --- A. PASS DETECTION ---------------------------------------------------
def assign_passes(df: pd.DataFrame) -> np.ndarray:
    """
    Label each message with a pass ID.

    Messages are grouped by satellite and ordered in time; a new pass begins
    whenever the gap to the previous message from that satellite exceeds
    PASS_GAP_SECONDS. Pass IDs are unique across satellites.

    If timestamps are unavailable, falls back to contiguous runs of
    global_index, which approximates the same thing because the dataset is
    stored in capture order.
    """
    if "meta_timestamp_global" in df.columns:
        raw = df["meta_timestamp_global"].to_numpy(dtype=float)

        # Identify the timestamp unit by magnitude. A current Unix epoch
        # value is ~1.7e9 in seconds, so each thousand-fold increase is one
        # step finer. Comparing raw nanosecond gaps against a threshold
        # expressed in seconds makes every gap look like a new pass, which
        # silently reduces GroupKFold to an ordinary split.
        median = float(np.nanmedian(np.abs(raw)))
        if median > 1e17:
            time_values, unit = raw / 1e9, "nanoseconds"
        elif median > 1e14:
            time_values, unit = raw / 1e6, "microseconds"
        elif median > 1e11:
            time_values, unit = raw / 1e3, "milliseconds"
        else:
            time_values, unit = raw, "seconds"
        source = f"timestamp ({unit} -> seconds)"
    else:
        time_values = df["global_index"].to_numpy(dtype=float)
        source = "global_index (timestamps unavailable)"

    pass_ids = np.empty(len(df), dtype=int)
    next_id = 0

    for sat in df["satellite_id"].unique():
        mask = (df["satellite_id"] == sat).to_numpy()
        idx = np.where(mask)[0]
        order = idx[np.argsort(time_values[idx])]

        current = next_id
        pass_ids[order[0]] = current
        for a, b in zip(order[:-1], order[1:]):
            gap = time_values[b] - time_values[a]
            threshold = (PASS_GAP_SECONDS if source.startswith("timestamp")
                         else 500)   # index units, not seconds
            if gap > threshold:
                current += 1
            pass_ids[b] = current
        next_id = current + 1

    print(f"  pass source: {source}")
    counts = pd.Series(pass_ids).value_counts()
    if counts.median() <= 1:
        print("  WARNING: every message became its own pass. Grouping is "
              "degenerate and GroupKFold will not differ from a random "
              "split. Check the timestamp unit.")
    return pass_ids


# --- A. GROUP-AWARE CROSS-VALIDATION -------------------------------------
def evaluate_cv(X: np.ndarray, y: np.ndarray,
                groups: np.ndarray | None,
                n_splits_override: int | None = None) -> dict:
    """
    Cross-validated accuracy, either stratified (random) or grouped by pass.

    With groups supplied, GroupKFold guarantees that all messages from one
    pass fall entirely within either the training or the test fold, never
    split across both.
    """
    model = lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=100,
                                         random_state=RANDOM_SEED, n_jobs=-1)),
    ])

    if groups is None:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_SEED)
        split_iter = splitter.split(X, y)
    else:
        n_groups = len(np.unique(groups))
        k = min(n_splits_override or N_SPLITS, n_groups)
        if k < 2:
            return {"mean": np.nan, "std": np.nan, "folds": [], "n_groups": n_groups}

        # StratifiedGroupKFold, not plain GroupKFold.
        #
        # Each pass belongs to exactly one satellite, so plain GroupKFold can
        # assign a test fold whose class distribution differs wildly from the
        # training folds. When that happens the majority-class baseline
        # collapses (observed: chance falling from 0.208 to 0.021), and the
        # model's accuracy is no longer comparable to it -- the evaluation
        # measures fold composition rather than classifier skill.
        #
        # StratifiedGroupKFold keeps whole passes together while balancing
        # class proportions across folds as far as the group structure allows.
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                        random_state=RANDOM_SEED)
        split_iter = splitter.split(X, y, groups)

    scores, chance = [], []
    for tr, te in split_iter:
        if len(np.unique(y[tr])) < 2:
            continue
        m = model().fit(X[tr], y[tr])
        scores.append(accuracy_score(y[te], m.predict(X[te])))
        d = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
        chance.append(accuracy_score(y[te], d.predict(X[te])))

    return {
        "mean":       float(np.mean(scores)) if scores else np.nan,
        "std":        float(np.std(scores))  if scores else np.nan,
        "chance":     float(np.mean(chance)) if chance else np.nan,
        "folds":      scores,
        "n_groups":   int(len(np.unique(groups))) if groups is not None else None,
    }


# --- B. McNEMAR'S TEST ---------------------------------------------------
def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    Exact McNemar test for two classifiers on the same test set.

    Builds the discordant counts:
        n01 = A wrong, B right
        n10 = A right, B wrong
    Under the null hypothesis that the two models are equally accurate,
    n10 given (n01 + n10) follows Binomial(n, 0.5). The exact binomial test
    is used rather than the chi-squared approximation, which is unreliable
    when the discordant count is small.

    Concordant cases (both right, or both wrong) carry no information about
    which model is better and are correctly excluded.
    """
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n_disc = n01 + n10

    if n_disc == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0,
                "significant": False,
                "note": "models made identical predictions"}

    p = float(scipy_stats.binomtest(n10, n_disc, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p_value": p,
            "significant": p < 0.05,
            "note": f"{n_disc} discordant predictions"}


def bootstrap_ci(correct: np.ndarray, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """95% percentile bootstrap CI for accuracy."""
    rng = np.random.default_rng(RANDOM_SEED)
    m = len(correct)
    scores = [correct[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# --- C. MAHALANOBIS SEPARABILITY -----------------------------------------
def mahalanobis_separability(df: pd.DataFrame,
                             features: list[str]) -> pd.DataFrame:
    """
    Pairwise Mahalanobis distance between satellite class means.

        D^2 = (mu_a - mu_b)^T  S^-1  (mu_a - mu_b)

    where S is the pooled within-class covariance. Unlike Euclidean
    distance, this discounts directions in which the classes are already
    highly variable, and it does not double-count correlated features.

    The pseudo-inverse is used because the feature set contains
    near-duplicates (std_I, var_I, iqr_I all measure spread), which makes
    the covariance matrix ill-conditioned.

    Interpretation: D is in units of pooled within-class standard
    deviations. D < 1 means the class means are closer together than the
    typical scatter within a class -- heavily overlapping, not separable.
    D > 3 would indicate well-separated classes.
    """
    X = df[features].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()
    sats = np.unique(y)

    # Pooled within-class covariance
    p = X.shape[1]
    pooled = np.zeros((p, p))
    dof = 0
    means = {}
    for s in sats:
        Xs = X[y == s]
        means[s] = Xs.mean(axis=0)
        Xc = Xs - means[s]
        pooled += Xc.T @ Xc
        dof += len(Xs) - 1
    pooled /= max(dof, 1)

    # Ridge term for numerical stability, then pseudo-inverse
    pooled += np.eye(p) * 1e-8 * np.trace(pooled) / p
    inv = np.linalg.pinv(pooled)

    rows = []
    for i, a in enumerate(sats):
        for b in sats[i + 1:]:
            d = means[a] - means[b]
            d2 = float(d @ inv @ d)
            rows.append({"sat_a": int(a), "sat_b": int(b),
                         "mahalanobis_D": float(np.sqrt(max(d2, 0.0))),
                         "mahalanobis_D2": d2})

    return (pd.DataFrame(rows)
            .sort_values("mahalanobis_D", ascending=False)
            .reset_index(drop=True))


# --- D. VARIANCE DECOMPOSITION OF `level` --------------------------------
def decompose_level_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute variation in `level` to candidate physical causes.

    For a continuous predictor (elevation) R^2 is the squared correlation.
    For a categorical predictor (beam / satellite) R^2 is the between-group
    sum of squares over the total sum of squares -- the fraction of variance
    explained by group membership alone.

    This addresses the circularity in the earlier analysis: showing that
    amplitude features correlate with an amplitude estimate proves little.
    What matters is whether `level` itself is driven by geometry.
    """
    if "meta_level" not in df.columns:
        return pd.DataFrame()

    level = df["meta_level"].to_numpy(dtype=float)
    ok = np.isfinite(level)
    rows = []

    def categorical_r2(labels: np.ndarray, values: np.ndarray) -> float:
        grand = values.mean()
        ss_total = float(np.sum((values - grand) ** 2))
        if ss_total == 0:
            return np.nan
        ss_between = 0.0
        for g in np.unique(labels):
            v = values[labels == g]
            if len(v):
                ss_between += len(v) * (v.mean() - grand) ** 2
        return ss_between / ss_total

    # Elevation (continuous)
    if "elevation_deg" in df.columns:
        m = ok & np.isfinite(df["elevation_deg"].to_numpy(float))
        if m.sum() > 10:
            r = float(np.corrcoef(df.loc[m, "elevation_deg"], level[m])[0, 1])
            rows.append({"predictor": "elevation angle",
                         "type": "continuous",
                         "r2": r ** 2,
                         "n": int(m.sum())})

    # Beam / cell (categorical)
    if "meta_ra_cell" in df.columns:
        cell = df["meta_ra_cell"].to_numpy()
        m = ok & pd.notna(cell)
        if m.sum() > 10:
            rows.append({"predictor": "beam (ra_cell)",
                         "type": "categorical",
                         "r2": categorical_r2(cell[m], level[m]),
                         "n": int(m.sum())})

    # Satellite identity (categorical) -- the upper bound on what an
    # ideal classifier could extract from `level` alone
    rows.append({"predictor": "satellite identity",
                 "type": "categorical",
                 "r2": categorical_r2(df["satellite_id"].to_numpy()[ok], level[ok]),
                 "n": int(ok.sum())})

    # Noise floor (continuous)
    if "meta_noise" in df.columns:
        noise = df["meta_noise"].to_numpy(float)
        m = ok & np.isfinite(noise)
        if m.sum() > 10:
            r = float(np.corrcoef(noise[m], level[m])[0, 1])
            rows.append({"predictor": "receiver noise floor",
                         "type": "continuous",
                         "r2": r ** 2,
                         "n": int(m.sum())})

    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("Statistical rigour checks")
    print("=" * 72)

    print("\nLoading...")
    df, features = load_all()
    X = df[features].to_numpy(dtype=float)
    y = df["satellite_id"].to_numpy()
    df = add_elevation(df)
    print(f"  {len(df):,} messages, {len(features)} features")

    # ---- A. Pass-aware evaluation ---------------------------------------
    print("\n" + "-" * 72)
    print("A. PASS-AWARE EVALUATION (GroupKFold)")
    print("-" * 72)
    passes = assign_passes(df)
    n_passes = len(np.unique(passes))
    sizes = pd.Series(passes).value_counts()
    print(f"  {n_passes:,} passes identified "
          f"(median {sizes.median():.0f} messages per pass)")

    random_cv = evaluate_cv(X, y, groups=None)

    # With few passes relative to classes, a grouped split can produce folds
    # whose class balance differs sharply from the training folds. When that
    # happens the majority-class baseline moves, and accuracies from the two
    # schemes are no longer on the same scale. Try several fold counts and
    # keep whichever produces a chance level closest to the random-split
    # baseline; report if none of them are usable.
    candidates = []
    for k in (5, 4, 3, 2):
        if k > len(np.unique(passes)):
            continue
        res = evaluate_cv(X, y, groups=passes, n_splits_override=k)
        if not np.isfinite(res.get("mean", np.nan)):
            continue
        res["n_splits"] = k
        res["chance_shift"] = abs(res["chance"] - random_cv["chance"])
        candidates.append(res)
        print(f"  {k}-fold grouped: acc={res['mean']:.4f}  "
              f"chance={res['chance']:.4f}  "
              f"chance shift={res['chance_shift']:+.4f}")

    group_cv = (min(candidates, key=lambda r: r["chance_shift"])
                if candidates else {"mean": np.nan, "std": np.nan,
                                    "chance": np.nan, "n_splits": None,
                                    "chance_shift": np.inf})
    if candidates:
        print(f"  using {group_cv['n_splits']}-fold "
              f"(smallest chance shift)\n")

    print(f"\n  Random  {N_SPLITS}-fold CV: {random_cv['mean']:.4f} "
          f"(+/- {random_cv['std']:.4f})   chance {random_cv['chance']:.4f}")
    print(f"  Grouped {N_SPLITS}-fold CV: {group_cv['mean']:.4f} "
          f"(+/- {group_cv['std']:.4f})   chance {group_cv['chance']:.4f}")
    gap = random_cv["mean"] - group_cv["mean"]
    print(f"  Optimism from pass leakage: {gap:+.4f}")

    # Guard: if the two chance levels differ materially, the folds are not
    # comparable and the difference above cannot be read as leakage.
    chance_shift = group_cv.get("chance_shift", np.inf)
    comparable = chance_shift <= 0.05
    if not comparable:
        print(f"\n  NOT INTERPRETABLE: the grouped chance level differs from "
              f"the random one by {chance_shift:.4f}.")
        print(f"  With only {n_passes} passes across "
              f"{len(np.unique(y))} satellites, no fold count produces "
              f"balanced groups.")
        print("  Pass-aware evaluation is underpowered on this subset and "
              "should be")
        print("  reported as a limitation rather than as a corrected result.")
        print(f"  (For reference only, margins over their own baselines: "
              f"random {random_cv['mean'] - random_cv['chance']:+.4f}, "
              f"grouped {group_cv['mean'] - group_cv['chance']:+.4f}.)")
    elif gap > 0.02:
        print("  => The random split was optimistic; some apparent skill was")
        print("     recognition of the pass rather than of the transmitter.")
    else:
        print("  => Negligible difference: pass leakage was not inflating the")
        print("     random-split result.")

    # ---- B. McNemar --------------------------------------------------
    print("\n" + "-" * 72)
    print("B. McNEMAR'S TEST (paired classifier comparison)")
    print("-" * 72)

    meta_cols = [f"meta_{c}" for c in METADATA_COLUMNS if f"meta_{c}" in df.columns]
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, np.arange(len(y)), test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    def fit_predict(Xa, Xb):
        m = Pipeline([("scale", StandardScaler()),
                      ("clf", RandomForestClassifier(
                          n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))])
        m.fit(Xa, ytr)
        return m.predict(Xb)

    pred_feat  = fit_predict(Xtr, Xte)
    pred_dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr).predict(Xte)

    correct_feat  = (pred_feat  == yte)
    correct_dummy = (pred_dummy == yte)

    results = []
    lo, hi = bootstrap_ci(correct_feat)
    results.append(("28 hand-crafted features", correct_feat.mean(), lo, hi))
    lo, hi = bootstrap_ci(correct_dummy)
    results.append(("Dummy (chance)", correct_dummy.mean(), lo, hi))

    correct_meta = None
    if meta_cols:
        Xm = df[meta_cols].to_numpy(dtype=float)
        for j in range(Xm.shape[1]):
            bad = ~np.isfinite(Xm[:, j])
            if bad.any():
                Xm[bad, j] = np.nanmedian(Xm[~bad, j]) if (~bad).any() else 0.0
        pred_meta = fit_predict(Xm[itr], Xm[ite])
        correct_meta = (pred_meta == yte)
        lo, hi = bootstrap_ci(correct_meta)
        results.append((f"{len(meta_cols)} channel metadata",
                        correct_meta.mean(), lo, hi))

    print(f"\n  {'Model':<28}{'Acc':>8}{'95% CI':>20}")
    print("  " + "-" * 56)
    for name, acc, lo, hi in results:
        print(f"  {name:<28}{acc:>8.4f}   [{lo:.4f}, {hi:.4f}]")

    print("\n  Paired tests against the chance baseline:")
    mc1 = mcnemar_test(correct_dummy, correct_feat)
    print(f"    28 features vs Dummy:  p = {mc1['p_value']:.4f}   "
          f"{'SIGNIFICANT' if mc1['significant'] else 'not significant'}  "
          f"({mc1['note']})")
    if correct_meta is not None:
        mc2 = mcnemar_test(correct_dummy, correct_meta)
        print(f"    metadata vs Dummy:     p = {mc2['p_value']:.4f}   "
              f"{'SIGNIFICANT' if mc2['significant'] else 'not significant'}  "
              f"({mc2['note']})")
        mc3 = mcnemar_test(correct_feat, correct_meta)
        print(f"    metadata vs features:  p = {mc3['p_value']:.4f}   "
              f"{'SIGNIFICANT' if mc3['significant'] else 'not significant'}  "
              f"({mc3['note']})")

    # ---- C. Mahalanobis ----------------------------------------------
    print("\n" + "-" * 72)
    print("C. MAHALANOBIS SEPARABILITY (replaces ad-hoc metric)")
    print("-" * 72)
    maha = mahalanobis_separability(df, features)
    print(f"\n  {'pair':<16}{'Mahalanobis D':>16}")
    print("  " + "-" * 32)
    for _, r in maha.head(5).iterrows():
        print(f"  {int(r['sat_a']):>3d} vs {int(r['sat_b']):<8d}"
              f"{r['mahalanobis_D']:>16.3f}")
    best = maha["mahalanobis_D"].max()
    print(f"\n  Best pairwise D = {best:.3f}")
    print("  Reference: D < 1 means class means are closer than the typical")
    print("  within-class scatter; D > 3 indicates well-separated classes.")

    # ---- D. Variance decomposition -----------------------------------
    print("\n" + "-" * 72)
    print("D. WHAT ACTUALLY DRIVES `level`?")
    print("-" * 72)
    decomp = decompose_level_variance(df)
    if len(decomp):
        print(f"\n  {'predictor':<26}{'type':<14}{'R^2':>8}")
        print("  " + "-" * 48)
        for _, r in decomp.iterrows():
            print(f"  {r['predictor']:<26}{r['type']:<14}{r['r2']:>8.3f}")
        print("\n  Earlier the claim was that features are dominated by the")
        print("  CHANNEL. What is actually established is that they are")
        print("  dominated by AMPLITUDE. The table above shows how much of")
        print("  the amplitude is attributable to geometry.")
    else:
        print("  level unavailable.")

    # ---- E. Within-beam classification -------------------------------
    print("\n" + "-" * 72)
    print("E. CLASSIFICATION WITHIN A SINGLE BEAM")
    print("-" * 72)
    beam_rows = []
    if "meta_ra_cell" in df.columns:
        counts = df["meta_ra_cell"].value_counts()
        for beam in counts.index[:3]:
            m = (df["meta_ra_cell"] == beam).to_numpy()
            if m.sum() < 300 or len(np.unique(y[m])) < 2:
                continue
            res = evaluate_cv(X[m], y[m], groups=None)
            beam_rows.append({"beam": beam, "n": int(m.sum()),
                              "accuracy": res["mean"], "chance": res["chance"]})
            print(f"  beam {beam}: n={int(m.sum()):>5,}  "
                  f"acc={res['mean']:.4f}  chance={res['chance']:.4f}")
        if not beam_rows:
            print("  No beam has enough messages for a stable estimate.")
    else:
        print("  ra_cell unavailable.")

    # ---- Outputs ------------------------------------------------------
    print("\n" + "-" * 72)
    print("Writing outputs...")
    pd.DataFrame([
        {"split": "random (StratifiedKFold)", "accuracy": random_cv["mean"],
         "std": random_cv["std"], "chance": random_cv["chance"]},
        {"split": "pass-aware (GroupKFold)",  "accuracy": group_cv["mean"],
         "std": group_cv["std"],  "chance": group_cv["chance"]},
    ]).to_csv(OUT_TABLES / "groupkfold_results.csv", index=False)
    maha.to_csv(OUT_TABLES / "mahalanobis_distances.csv", index=False)
    if len(decomp):
        decomp.to_csv(OUT_TABLES / "level_variance_decomposition.csv", index=False)

    md = f"""# Statistical rigour checks

Five corrections to the earlier analysis.

## A. Pass-aware evaluation

Messages captured during the same satellite pass share channel conditions.
A random train/test split can place messages from one pass on both sides of
the split, allowing a model to score above chance by recognising the pass
rather than the transmitter. Passes were recovered from gaps in the capture
timestamps ({n_passes:,} passes identified) and used as cross-validation
groups so that no pass is ever split.

| Split | Accuracy | Chance |
|-------|---------:|-------:|
| Random (StratifiedKFold) | {random_cv['mean']:.4f} ± {random_cv['std']:.4f} | {random_cv['chance']:.4f} |
| Pass-aware (GroupKFold) | {group_cv['mean']:.4f} ± {group_cv['std']:.4f} | {group_cv['chance']:.4f} |

Difference: {gap:+.4f}.

## B. Paired classifier comparison

Accuracy point estimates were previously used to describe one model as
performing better than another. Because all models are evaluated on the
same test set their errors are paired, and the correct test is McNemar's,
which considers only the discordant predictions.

| Model | Accuracy | 95% CI (bootstrap) |
|-------|---------:|:------------------:|
"""
    for name, acc, lo, hi in results:
        md += f"| {name} | {acc:.4f} | [{lo:.4f}, {hi:.4f}] |\n"

    md += f"""
McNemar, 28 features vs chance: p = {mc1['p_value']:.4f}
({'significant' if mc1['significant'] else 'not significant'} at alpha = 0.05).
"""
    if correct_meta is not None:
        md += f"""
McNemar, channel metadata vs chance: p = {mc2['p_value']:.4f}
({'significant' if mc2['significant'] else 'not significant'}).

McNemar, metadata vs 28 features: p = {mc3['p_value']:.4f}
({'significant' if mc3['significant'] else 'not significant'}).
"""

    md += f"""
## C. Mahalanobis separability

The earlier separability figure used Euclidean distance between class means
with an ad-hoc normalisation and an invented threshold. Euclidean distance
is unsuitable here because the feature set contains near-duplicates
(`std_I`, `var_I` and `iqr_I` all measure spread), so correlated quantities
are counted repeatedly. Mahalanobis distance uses the pooled within-class
covariance and is the standard measure.

Best pairwise Mahalanobis distance: **D = {best:.3f}**

D is expressed in pooled within-class standard deviations. D below 1
indicates that class means are closer together than the typical scatter
within a class; D above 3 would indicate well-separated classes.

## D. What drives `level`

The earlier analysis reported that many features correlate with `level` at
R² above 0.98 and described this as channel dominance. That inference was
partly circular: `level` is itself an estimate of received amplitude, so any
amplitude-derived feature must correlate with it. The substantive question
is what drives `level`.

| Predictor | Type | R² |
|-----------|------|---:|
"""
    for _, r in decomp.iterrows():
        md += f"| {r['predictor']} | {r['type']} | {r['r2']:.3f} |\n"

    md += """
The defensible claim is therefore that the hand-crafted features are
dominated by **signal amplitude**. The extent to which that amplitude is
attributable to orbital geometry is given by the table above and should be
stated as measured rather than assumed.

## E. Within-beam classification

Iridium satellites transmit through multiple spot beams with different
antenna patterns. Holding the beam constant is a tighter channel control
than holding elevation constant, because it fixes the antenna gain toward
the receiver as well as the approximate geometry.

"""
    if beam_rows:
        md += "| Beam | Messages | Accuracy | Chance |\n|---|---:|---:|---:|\n"
        for r in beam_rows:
            md += (f"| {r['beam']} | {r['n']:,} | "
                   f"{r['accuracy']:.4f} | {r['chance']:.4f} |\n")
    else:
        md += "Insufficient messages per beam for a stable estimate.\n"

    (OUT_REPORTS / "statistical_rigour.md").write_text(md)
    print("  outputs/tables/groupkfold_results.csv")
    print("  outputs/tables/mahalanobis_distances.csv")
    print("  outputs/tables/level_variance_decomposition.csv")
    print("  outputs/reports/statistical_rigour.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
