"""Test whether specific burst regions carry more discriminative power."""
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

DATA_DIR = Path("data/raw")
TARGET_SATS = [51, 85, 87, 92, 109]
MAX_VALID_TS = 5e15

# Load session A
samples = np.load(DATA_DIR / "samples_000.npy")
sats = np.load(DATA_DIR / "ra_sat_000.npy")
ts = np.load(DATA_DIR / "timestamp_000.npy")
mask = np.isin(sats, TARGET_SATS) & (ts < MAX_VALID_TS)
X_raw = samples[mask]  # (N, 11000, 2)
y_sat = sats[mask]
sat_to_idx = {s: i for i, s in enumerate(TARGET_SATS)}
y = np.array([sat_to_idx[s] for s in y_sat])

print(f"Loaded {len(X_raw)} bursts\n")

def extract_stats(iq_region):
    """Simple stats from an IQ region: mean, std, skew, kurt per channel + cross."""
    feats = []
    for burst in iq_region:
        I, Q = burst[:, 0], burst[:, 1]
        amp = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        f = [
            I.mean(), I.std(), Q.mean(), Q.std(),
            amp.mean(), amp.std(),
            phase.mean(), phase.std(),
            np.corrcoef(I, Q)[0, 1],
            np.mean(I**3) / (I.std()**3 + 1e-10),  # skewness I
            np.mean(Q**3) / (Q.std()**3 + 1e-10),  # skewness Q
            np.mean(I**4) / (I.std()**4 + 1e-10),  # kurtosis I
            np.mean(Q**4) / (Q.std()**4 + 1e-10),  # kurtosis Q
        ]
        feats.append(f)
    return np.array(feats)

# Define regions to test
regions = {
    "full_burst (0:11000)":     (0, 11000),
    "first_100 (transient)":    (0, 100),
    "first_250 (transient)":    (0, 250),
    "first_500 (transient)":    (0, 500),
    "first_1000":               (0, 1000),
    "mid_burst (4000:7000)":    (4000, 7000),
    "last_500 (tail)":          (10500, 11000),
    "first_100 + last_100":     None,  # special case
}

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X_raw, y))

print(f"{'Region':<30} {'Train acc':>10} {'Test acc':>10}")
print("-" * 52)

for name, bounds in regions.items():
    if bounds is not None:
        start, end = bounds
        X_region = X_raw[:, start:end, :]
    else:
        # Concatenate first and last 100
        X_region = np.concatenate([X_raw[:, :100, :], X_raw[:, -100:, :]], axis=1)

    feats = extract_stats(X_region)
    feats = np.nan_to_num(feats)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(feats[train_idx], y[train_idx])
    tr_acc = accuracy_score(y[train_idx], rf.predict(feats[train_idx]))
    te_acc = accuracy_score(y[test_idx], rf.predict(feats[test_idx]))
    print(f"{name:<30} {tr_acc:>10.3f} {te_acc:>10.3f}")

# Also test: raw flattened IQ for short regions (CNN-like but with RF)
print(f"\n{'--- Raw flattened IQ (no feature engineering) ---':^52}")
print(f"{'Region':<30} {'Train acc':>10} {'Test acc':>10}")
print("-" * 52)

for name, bounds in [("first_100 raw", (0, 100)), ("first_250 raw", (0, 250)),
                       ("first_500 raw", (0, 500))]:
    start, end = bounds
    X_flat = X_raw[:, start:end, :].reshape(len(X_raw), -1)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_flat[train_idx], y[train_idx])
    tr_acc = accuracy_score(y[train_idx], rf.predict(X_flat[train_idx]))
    te_acc = accuracy_score(y[test_idx], rf.predict(X_flat[test_idx]))
    print(f"{name:<30} {tr_acc:>10.3f} {te_acc:>10.3f}")

print("\nIf ANY region shows test_acc >> 20%, that's where the fingerprint lives.")
print("If nothing works, the signal genuinely lacks hardware discrimination.")
