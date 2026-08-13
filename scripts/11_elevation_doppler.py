"""
06_elevation_doppler.py
=======================

PURPOSE
-------

  STEP 12 - ELEVATION AND DOPPLER
    The dataset has no elevation column, but it does contain the satellite's
    latitude, longitude and altitude at the moment of capture. Combined with
    the known receiver location (Oxford), the elevation angle can be computed
    directly. This script:
      1. Computes elevation angle per message from satellite geometry.
      2. Tests whether `level` (received signal strength) tracks elevation --
         confirming the physical mechanism behind the channel-dominance result.
      3. Splits messages into low (0-30 deg), medium (30-60) and high (60-90)
         elevation bands and re-trains a classifier within each band.
      4. Examines centre frequency for Doppler structure.

    WHY THE BAND EXPERIMENT IS DECISIVE
    Within a single elevation band, channel conditions are far more uniform.
    If accuracy STAYS at chance inside each band, the channel confound has
    been controlled and the features demonstrably carry no transmitter
    information. If accuracy RISES, hardware signal was being masked by
    channel variation. Either outcome is a real finding.

  STEP 11 - SATELLITE SEPARABILITY
    Computes the mean feature vector per satellite and the pairwise distances
    between them, to identify which satellites are furthest apart in feature
    space.

    Distances between means are reported BOTH raw and normalised by
    within-class spread. Raw distance alone is misleading: if the scatter
    within each satellite is larger than the gap between satellites, the
    classes are not actually separable no matter how far apart their centres
    appear. The normalised figure is the one that matters.

    The script then checks whether any apparent separation is explained by
    differing elevation distributions -- i.e. whether "separable in feature
    space" really means "observed at different geometries".

USAGE
-----
    python scripts/06_elevation_doppler.py

OUTPUTS
-------
    outputs/figures/elevation_analysis.png
    outputs/figures/satellite_separability.png
    outputs/tables/elevation_bands.csv
    outputs/tables/satellite_distances.csv
    outputs/reports/elevation_doppler.md
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as scipy_stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# --- PATHS ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "outputs" / "tables" / "features.csv"
OUT_TABLES   = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS  = PROJECT_ROOT / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)


# --- CONFIG ---------------------------------------------------------------
# Receiver: rooftop antenna at the University of Oxford (Smailes et al. 2023).
# The paper does not give exact coordinates; these are Oxford city centre.
# Elevation angle is insensitive to errors of a few hundred metres when the
# satellite is ~780 km away, so this approximation is acceptable. State it
# explicitly in the dissertation.
RECEIVER_LAT_DEG = 51.7548
RECEIVER_LON_DEG = -1.2544
RECEIVER_ALT_M   = 60.0

RANDOM_SEED   = 42
TEST_FRACTION = 0.20
MIN_BAND_SIZE = 200        # minimum messages needed to train within a band

NON_FEATURE_COLUMNS = {"sample_id", "global_index", "index",
                       "Unnamed: 0", "satellite_id"}

# WGS84 ellipsoid constants
WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# --- STEP 1: LOAD --------------------------------------------------------
def load_metadata_column(column: str) -> np.ndarray | None:
    files = sorted(DATA_RAW.glob(f"{column}_*.npy"))
    if not files:
        return None
    return np.concatenate([np.load(f) for f in files])


def load_everything() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    index_col = next((c for c in ("global_index", "sample_id")
                      if c in df.columns), None)
    if index_col is None:
        raise KeyError("No global_index / sample_id column to join metadata on.")
    rows = df[index_col].to_numpy(dtype=int)

    for column in ["level", "noise", "ra_lat", "ra_lon", "ra_alt", "ra_cell",
                   "center_frequency", "timestamp_global"]:
        full = load_metadata_column(column)
        if full is None or rows.max() >= len(full):
            print(f"    {column:<18s} unavailable")
            continue
        df[f"meta_{column}"] = full[rows]
        print(f"    {column:<18s} attached")

    return df, feature_names


# --- STEP 2: UNIT DETECTION ----------------------------------------------
def detect_angle_units(values: np.ndarray, name: str) -> np.ndarray:
    """
    Return angles in DEGREES.

    Latitude in degrees spans -90..90; in radians it spans -1.57..1.57.
    A maximum absolute value above 2 therefore indicates degrees.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"{name} has no finite values")

    if np.max(np.abs(finite)) > 2.0:
        print(f"    {name}: detected DEGREES "
              f"(range {finite.min():.2f} to {finite.max():.2f})")
        return values
    print(f"    {name}: detected RADIANS "
          f"(range {finite.min():.4f} to {finite.max():.4f}) -> converting")
    return np.degrees(values)


def interpret_radial_column(values: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Return the satellite's GEOCENTRIC RADIUS in metres, plus a description.

    `ra_alt` may be stored either as height above the surface or as distance
    from the centre of the Earth, in metres or kilometres. Getting this wrong
    silently corrupts every elevation angle, so the value is identified by
    comparing it against known physical scales:

        Earth mean radius            ~ 6,371 km
        Iridium orbital altitude     ~   780 km
        Iridium geocentric radius    ~ 7,151 km

    A median near 7,100 therefore indicates geocentric radius in kilometres;
    a median near 780 indicates altitude above the surface.
    """
    EARTH_RADIUS_M = 6_371_000.0
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))

    if 6_000 < median < 10_000:                    # kilometres, geocentric
        radius_m = values * 1000.0
        desc = (f"geocentric radius in KILOMETRES (median {median:,.1f} km "
                f"= {median - 6371:,.0f} km altitude)")
    elif 6_000_000 < median < 10_000_000:          # metres, geocentric
        radius_m = values
        desc = (f"geocentric radius in METRES (median {median:,.0f} m "
                f"= {(median - EARTH_RADIUS_M)/1000:,.0f} km altitude)")
    elif 100 < median < 2_000:                     # kilometres, altitude
        radius_m = values * 1000.0 + EARTH_RADIUS_M
        desc = f"altitude above surface in KILOMETRES (median {median:,.1f} km)"
    elif 100_000 < median < 2_000_000:             # metres, altitude
        radius_m = values + EARTH_RADIUS_M
        desc = f"altitude above surface in METRES (median {median:,.0f} m)"
    else:
        radius_m = values
        desc = f"UNRECOGNISED scale (median {median:,.2f}) - treating as metres radius"

    return radius_m, desc


# --- POSITION FIELD VALIDATION -------------------------------------------
EARTH_RADIUS_KM   = 6371.0
ORBITAL_SPEED_KMS = 7.5        # Iridium, low Earth orbit
PASS_GAP_S        = 20 * 60


def great_circle_km(lat1, lon1, lat2, lon2):
    """Distance along the Earth's surface between two sub-points, in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def validate_position_fields(df: pd.DataFrame,
                             lat: np.ndarray,
                             lon: np.ndarray) -> dict:
    """
    Check that ra_lat / ra_lon actually describe the satellite named by
    ra_sat, BEFORE deriving elevation from them.

    Elevation angle is only meaningful if the position fields track the
    transmitting satellite. Computing it from fields that describe
    something else produces numbers that look plausible and mean nothing,
    which is worse than producing none. This function therefore acts as a
    precondition: the elevation analysis runs only if it passes.

    TEST - ORBITAL CONSISTENCY
      A satellite in low Earth orbit moves at about 7.5 km/s, so between
      consecutive messages dt seconds apart its sub-point can travel at
      most roughly 7.5 * dt km. Implied speeds are computed for consecutive
      messages from the same satellite within a single pass.

    DIAGNOSIS - IF THE TEST FAILS
      Positions are compared under two groupings: by satellite and by beam
      (ra_cell). Each Iridium satellite illuminates 48 spot beams across a
      footprint thousands of kilometres wide, so if the fields are
      beam-referenced rather than satellite-referenced, positions will
      cluster far more tightly by beam index than by satellite identity.

    Returns a dictionary with the verdict and supporting numbers.
    """
    result = {"tested": False, "valid": False}

    if "meta_timestamp_global" not in df.columns:
        result["reason"] = "no timestamps available"
        return result

    ts = df["meta_timestamp_global"].to_numpy(dtype=float)
    if np.nanmedian(np.abs(ts)) > 1e17:
        ts = ts / 1e9                      # nanoseconds -> seconds

    sat = df["satellite_id"].to_numpy()
    speeds = []

    for s in np.unique(sat):
        m = np.where(sat == s)[0]
        if len(m) < 20:
            continue
        order = m[np.argsort(ts[m])]
        t = ts[order]
        dt = np.diff(t)
        keep = (dt > 0) & (dt < PASS_GAP_S)    # within one pass only
        if keep.sum() < 5:
            continue
        d = great_circle_km(lat[order][:-1][keep], lon[order][:-1][keep],
                            lat[order][1:][keep],  lon[order][1:][keep])
        speeds.append(d / dt[keep])

    if not speeds:
        result["reason"] = "insufficient within-pass message pairs"
        return result

    speeds = np.concatenate(speeds)
    # Allow 50% headroom for timing jitter and coordinate quantisation
    frac_ok = float((speeds <= ORBITAL_SPEED_KMS * 1.5).mean())

    result.update({
        "tested": True,
        "median_speed_kms": float(np.median(speeds)),
        "frac_plausible": frac_ok,
        "n_pairs": int(len(speeds)),
        "valid": frac_ok > 0.90,
    })

    # If the fields fail, work out what they describe instead
    if not result["valid"] and "meta_ra_cell" in df.columns:
        cell = df["meta_ra_cell"].to_numpy()

        def median_scatter(groups):
            out = []
            for g in np.unique(groups):
                m = (groups == g)
                if m.sum() < 20:
                    continue
                out.append(np.median(great_circle_km(
                    lat[m], lon[m], np.median(lat[m]), np.median(lon[m]))))
            return float(np.median(out)) if out else np.nan

        result["scatter_by_satellite"] = median_scatter(sat)
        result["scatter_by_beam"]      = median_scatter(cell)
        result["beam_explains"] = bool(
            np.isfinite(result["scatter_by_beam"])
            and np.isfinite(result["scatter_by_satellite"])
            and result["scatter_by_beam"] < result["scatter_by_satellite"] * 0.6)

    return result


# --- STEP 3: ELEVATION ANGLE ---------------------------------------------
def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """Convert geodetic coordinates to Earth-Centred Earth-Fixed metres."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + alt_m) * np.sin(lat)
    return x, y, z


def elevation_angle_deg(sat_lat_deg, sat_lon_deg, sat_radius_m) -> np.ndarray:
    """
    Elevation angle of the satellite as seen from the receiver, in degrees.

    The satellite position is given as geocentric latitude/longitude and
    distance from the Earth's centre, so it converts to ECEF directly by
    spherical trigonometry -- NOT via the geodetic formula, which would add
    the Earth's radius a second time.

    Method: receiver->satellite vector in ECEF, rotated into the receiver's
    local East-North-Up frame; elevation = atan2(Up, horizontal distance).

    90 degrees = directly overhead, 0 = on the horizon, negative = below it
    (physically impossible for a received message, so negative values
    indicate a corrupted position field).
    """
    rx = np.array(geodetic_to_ecef(RECEIVER_LAT_DEG,
                                   RECEIVER_LON_DEG,
                                   RECEIVER_ALT_M))

    lat = np.radians(sat_lat_deg)
    lon = np.radians(sat_lon_deg)
    sx = sat_radius_m * np.cos(lat) * np.cos(lon)
    sy = sat_radius_m * np.cos(lat) * np.sin(lon)
    sz = sat_radius_m * np.sin(lat)

    dx, dy, dz = sx - rx[0], sy - rx[1], sz - rx[2]

    lat0 = np.radians(RECEIVER_LAT_DEG)
    lon0 = np.radians(RECEIVER_LON_DEG)

    east  = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = (-np.sin(lat0) * np.cos(lon0) * dx
             - np.sin(lat0) * np.sin(lon0) * dy
             + np.cos(lat0) * dz)
    up    = ( np.cos(lat0) * np.cos(lon0) * dx
             + np.cos(lat0) * np.sin(lon0) * dy
             + np.sin(lat0) * dz)

    horizontal = np.sqrt(east ** 2 + north ** 2)
    return np.degrees(np.arctan2(up, horizontal))


# --- STEP 4: PER-BAND CLASSIFICATION -------------------------------------
def classify_within_band(X: np.ndarray, y: np.ndarray) -> dict | None:
    """Train a Random Forest on one elevation band; return accuracy vs chance."""
    if len(y) < MIN_BAND_SIZE or len(np.unique(y)) < 2:
        return None

    # Every class must have at least 2 members for a stratified split
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION,
        random_state=RANDOM_SEED, stratify=y)

    forest = RandomForestClassifier(n_estimators=200,
                                    random_state=RANDOM_SEED, n_jobs=-1)
    forest.fit(X_tr, y_tr)
    accuracy = accuracy_score(y_te, forest.predict(X_te))

    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    chance = accuracy_score(y_te, dummy.predict(X_te))

    # Bootstrap CI so bands with few samples are not over-interpreted
    rng = np.random.default_rng(RANDOM_SEED)
    y_pred = forest.predict(X_te)
    boot = [accuracy_score(y_te[i], y_pred[i])
            for i in (rng.integers(0, len(y_te), len(y_te)) for _ in range(500))]

    return {
        "n_messages": len(y),
        "n_test":     len(y_te),
        "accuracy":   accuracy,
        "ci_low":     float(np.percentile(boot, 2.5)),
        "ci_high":    float(np.percentile(boot, 97.5)),
        "chance":     chance,
        "n_classes":  len(np.unique(y)),
    }


# --- STEP 5: SATELLITE SEPARABILITY (STEP 11) ----------------------------
def satellite_separability(df: pd.DataFrame,
                           feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Distance between satellites in standardised feature space.

    Two measures are returned:

      raw_distance        Euclidean distance between the two satellites'
                          mean feature vectors.

      normalised_distance The same distance divided by the pooled within-class
                          standard deviation. This is the measure that matters:
                          a large gap between centres means nothing if the
                          scatter inside each class is larger still.
                          Values below ~1 indicate heavily overlapping classes.
    """
    X = df[feature_names].to_numpy(dtype=float)
    # Standardise so every feature contributes comparably to the distance
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y = df["satellite_id"].to_numpy()
    sats = np.unique(y)

    means, spreads = {}, {}
    for s in sats:
        Xs = X[y == s]
        means[s]   = Xs.mean(axis=0)
        spreads[s] = Xs.std(axis=0)

    raw  = pd.DataFrame(index=sats, columns=sats, dtype=float)
    norm = pd.DataFrame(index=sats, columns=sats, dtype=float)

    for a in sats:
        for b in sats:
            if a == b:
                raw.loc[a, b] = 0.0
                norm.loc[a, b] = 0.0
                continue
            diff = means[a] - means[b]
            raw.loc[a, b] = float(np.linalg.norm(diff))
            # Pooled within-class spread, per feature, then combined
            pooled = np.sqrt((spreads[a] ** 2 + spreads[b] ** 2) / 2.0)
            norm.loc[a, b] = float(np.linalg.norm(diff / (pooled + 1e-12))
                                   / np.sqrt(len(feature_names)))

    return raw, norm


# --- STEP 6: PLOTS -------------------------------------------------------
def plot_elevation(df: pd.DataFrame, bands: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Elevation distribution
    ax = axes[0, 0]
    ax.hist(df["elevation_deg"].dropna(), bins=50, color="steelblue")
    for edge in (30, 60):
        ax.axvline(edge, color="crimson", ls="--", lw=1.2)
    ax.set_xlabel("Elevation angle (degrees)")
    ax.set_ylabel("Messages")
    ax.set_title("Elevation distribution (bands marked)", fontweight="bold")
    ax.grid(alpha=0.3)

    # level vs elevation -- the physical mechanism
    ax = axes[0, 1]
    ok = df["elevation_deg"].notna() & df["meta_level"].notna()
    ax.scatter(df.loc[ok, "elevation_deg"], df.loc[ok, "meta_level"],
               s=3, alpha=0.25, color="darkorange")
    if ok.sum() > 10:
        r = np.corrcoef(df.loc[ok, "elevation_deg"],
                        df.loc[ok, "meta_level"])[0, 1]
        ax.set_title(f"Signal level vs elevation  (r = {r:+.3f})",
                     fontweight="bold")
    ax.set_xlabel("Elevation angle (degrees)")
    ax.set_ylabel("level (received signal strength)")
    ax.grid(alpha=0.3)

    # Elevation distribution per satellite
    ax = axes[1, 0]
    sats = sorted(df["satellite_id"].unique())
    data = [df.loc[df["satellite_id"] == s, "elevation_deg"].dropna()
            for s in sats]
    ax.boxplot(data, labels=[str(s) for s in sats])
    ax.set_xlabel("Satellite ID")
    ax.set_ylabel("Elevation angle (degrees)")
    ax.set_title("Elevation distribution per satellite\n"
                 "(differences here are geometry, not hardware)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Accuracy per band
    ax = axes[1, 1]
    if len(bands):
        x = np.arange(len(bands))
        ax.bar(x - 0.2, bands["accuracy"], 0.4, label="Random Forest",
               color="steelblue",
               yerr=[bands["accuracy"] - bands["ci_low"],
                     bands["ci_high"] - bands["accuracy"]],
               capsize=4)
        ax.bar(x + 0.2, bands["chance"], 0.4, label="Chance", color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(bands["band"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification accuracy within elevation bands",
                     fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_separability(raw: pd.DataFrame, norm: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mat, title in [
        (ax1, raw,  "Raw distance between satellite mean vectors"),
        (ax2, norm, "Distance normalised by within-class spread\n"
                    "(<1 means classes overlap heavily)"),
    ]:
        values = mat.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="viridis")
        ax.set_xticks(range(len(mat)))
        ax.set_yticks(range(len(mat)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticklabels(mat.index)
        ax.set_title(title, fontweight="bold", fontsize=11)
        for i in range(len(mat)):
            for j in range(len(mat)):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --- MAIN ----------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Elevation / Doppler analysis and satellite separability")
    print("=" * 70)

    print("\nStep 1 - Loading features and metadata...")
    df, feature_names = load_everything()
    print(f"  Messages: {len(df):,}   Features: {len(feature_names)}")

    # --- Elevation -------------------------------------------------------
    have_geometry = all(f"meta_{c}" in df.columns
                        for c in ("ra_lat", "ra_lon", "ra_alt"))
    bands_table = pd.DataFrame()

    if not have_geometry:
        print("\nStep 2 - SKIPPED: satellite position columns unavailable.")
        df["elevation_deg"] = np.nan
    else:
        print("\nStep 2 - Detecting metadata units...")
        lat = detect_angle_units(df["meta_ra_lat"].to_numpy(float), "ra_lat")
        lon = detect_angle_units(df["meta_ra_lon"].to_numpy(float), "ra_lon")
        radius_m, radius_desc = interpret_radial_column(
            df["meta_ra_alt"].to_numpy(float))
        print(f"    ra_alt: {radius_desc}")

        # ---- Validate the position fields BEFORE using them -----------
        print("\nStep 3 - Validating position fields against orbital motion...")
        check = validate_position_fields(df, lat, lon)

        if not check["tested"]:
            print(f"  Could not test: {check.get('reason', 'unknown')}")
            print("  Proceeding, but elevation results are unverified.")
        else:
            print(f"  Consecutive within-pass message pairs: "
                  f"{check['n_pairs']:,}")
            print(f"  Median implied sub-point speed: "
                  f"{check['median_speed_kms']:.2f} km/s")
            print(f"  Iridium orbital velocity:       "
                  f"{ORBITAL_SPEED_KMS:.2f} km/s")
            print(f"  Pairs within 1.5x orbital velocity: "
                  f"{check['frac_plausible']:.1%}")

        if check.get("tested") and not check["valid"]:
            print("\n  FAILED. The position fields do not track the satellite")
            print("  named by ra_sat, so elevation angle cannot be derived")
            print("  from them. The elevation analysis is SKIPPED rather than")
            print("  reported with invalid values.")

            if "scatter_by_beam" in check:
                print(f"\n  Diagnosis - median positional scatter within:")
                print(f"    one satellite : "
                      f"{check['scatter_by_satellite']:>10,.0f} km")
                print(f"    one beam      : "
                      f"{check['scatter_by_beam']:>10,.0f} km")
                if check.get("beam_explains"):
                    print("\n  Positions cluster far more tightly by beam than")
                    print("  by satellite, indicating these are beam reference")
                    print("  positions rather than satellite positions.")
                else:
                    print("\n  Beam index does not explain the positions either.")
                    print("  The referent of these fields is undetermined.")

            df["elevation_deg"] = np.nan
            have_geometry = False
        else:
            print("\nStep 3b - Computing elevation angle from satellite geometry...")
            print(f"  Receiver: {RECEIVER_LAT_DEG:.4f} N, "
                  f"{RECEIVER_LON_DEG:.4f} E, {RECEIVER_ALT_M:.0f} m")
            df["elevation_deg"] = elevation_angle_deg(lat, lon, radius_m)

            below = (df["elevation_deg"] < 0).sum()
            frac  = below / len(df)
            print(f"  Messages with elevation < 0 (impossible): "
                  f"{below:,} ({frac:.1%})")
            df.loc[df["elevation_deg"] < 0, "elevation_deg"] = np.nan
            el = df["elevation_deg"].dropna()
            if len(el):
                print(f"  Elevation range: {el.min():.1f} to "
                      f"{el.max():.1f} degrees")
                print(f"  Median: {el.median():.1f} degrees")

        # Does level track elevation? This is the physical mechanism.
        # Only meaningful if the elevation values are themselves valid.
        if have_geometry and "meta_level" in df.columns:
            ok = df["elevation_deg"].notna() & df["meta_level"].notna()
            r   = float(np.corrcoef(df.loc[ok, "elevation_deg"],
                                    df.loc[ok, "meta_level"])[0, 1])
            rho = float(scipy_stats.spearmanr(df.loc[ok, "elevation_deg"],
                                              df.loc[ok, "meta_level"]).statistic)
            print(f"\n  level vs elevation:  Pearson r = {r:+.3f}   "
                  f"Spearman rho = {rho:+.3f}")
            if abs(rho) > 0.3:
                print("  => Signal level tracks elevation, as physics predicts.")

        # --- Band experiment (Step 12) -----------------------------------
        if not have_geometry:
            print("\nStep 4 - SKIPPED: elevation could not be validated, so")
            print("  banding by elevation would partition on a quantity that")
            print("  does not describe the observation geometry.")
        else:
            print("\nStep 4 - Classification within elevation bands...")
            X = df[feature_names].to_numpy(dtype=float)
            y = df["satellite_id"].to_numpy()
            rows = []
            for label, lo, hi in [("Low 0-30", 0, 30),
                                  ("Medium 30-60", 30, 60),
                                  ("High 60-90", 60, 90)]:
                mask = ((df["elevation_deg"] >= lo)
                        & (df["elevation_deg"] < hi)).to_numpy()
                res = classify_within_band(X[mask], y[mask])
                if res is None:
                    print(f"  {label:<14s} {int(mask.sum()):>5,} messages "
                          f"-- too few to train")
                    continue
                rows.append({"band": label, **res})
                print(f"  {label:<14s} {res['n_messages']:>5,} messages   "
                      f"acc={res['accuracy']:.4f} "
                      f"CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}]   "
                      f"chance={res['chance']:.4f}")
            bands_table = pd.DataFrame(rows)

    # --- Doppler ---------------------------------------------------------
    if "meta_center_frequency" in df.columns:
        print("\nStep 5 - Centre-frequency (Doppler) inspection...")
        cf = df["meta_center_frequency"].to_numpy(float)
        print(f"  Overall: median {np.median(cf):,.1f} Hz, "
              f"spread {np.std(cf):,.1f} Hz")
        print("  Per satellite:")
        for s in sorted(df["satellite_id"].unique()):
            sub = cf[df["satellite_id"].to_numpy() == s]
            print(f"    Sat {int(s):>3d}: median {np.median(sub):>14,.1f}   "
                  f"std {np.std(sub):>12,.1f}")
        if have_geometry and df["elevation_deg"].notna().any():
            ok = df["elevation_deg"].notna() & np.isfinite(cf)
            rho = float(scipy_stats.spearmanr(
                df.loc[ok, "elevation_deg"], cf[ok.to_numpy()]).statistic)
            print(f"  centre frequency vs elevation: Spearman rho = {rho:+.3f}")

    # --- Separability (Step 11) -----------------------------------------
    print("\nStep 6 - Satellite separability in feature space (Step 11)...")
    raw, norm = satellite_separability(df, feature_names)

    pairs = []
    sats = list(raw.index)
    for i, a in enumerate(sats):
        for b in sats[i + 1:]:
            pairs.append({"sat_a": a, "sat_b": b,
                          "raw_distance": raw.loc[a, b],
                          "normalised_distance": norm.loc[a, b]})
    pairs_df = (pd.DataFrame(pairs)
                .sort_values("normalised_distance", ascending=False)
                .reset_index(drop=True))

    print("  Most separated pairs (normalised by within-class spread):")
    for _, r in pairs_df.head(5).iterrows():
        print(f"    Sat {int(r['sat_a']):>3d} vs {int(r['sat_b']):>3d}   "
              f"raw={r['raw_distance']:.3f}   "
              f"normalised={r['normalised_distance']:.3f}")

    max_norm = pairs_df["normalised_distance"].max()
    if max_norm < 1.0:
        print(f"  => Best normalised separation is {max_norm:.3f} (<1).")
        print("     Within-class scatter exceeds between-class distance for")
        print("     every pair: no subset of satellites is cleanly separable.")

    # Is apparent separation explained by elevation?
    if have_geometry and df["elevation_deg"].notna().any():
        print("\n  Median elevation per satellite:")
        for s in sorted(df["satellite_id"].unique()):
            med = df.loc[df["satellite_id"] == s, "elevation_deg"].median()
            print(f"    Sat {int(s):>3d}: {med:6.1f} degrees")

    # --- Outputs ---------------------------------------------------------
    print("\nStep 7 - Writing outputs...")
    if len(bands_table):
        bands_table.to_csv(OUT_TABLES / "elevation_bands.csv", index=False)
        print("  outputs/tables/elevation_bands.csv")
    pairs_df.to_csv(OUT_TABLES / "satellite_distances.csv", index=False)
    print("  outputs/tables/satellite_distances.csv")

    if have_geometry and df["elevation_deg"].notna().any():
        plot_elevation(df, bands_table, OUT_FIGURES / "elevation_analysis.png")
        print("  outputs/figures/elevation_analysis.png")
    plot_separability(raw, norm, OUT_FIGURES / "satellite_separability.png")
    print("  outputs/figures/satellite_separability.png")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
