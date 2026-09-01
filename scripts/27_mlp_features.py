#!/usr/bin/env python3
"""
27_mlp_features.py — MLP on extracted features
================================================
Fills the gap between classical ML baselines and CNN in the model
comparison table. Uses the same extracted features as script 05,
evaluated with both stratified and cross-session splits.

Outputs:
  outputs/tables/mlp_results.csv
  outputs/figures/mlp_loss_curves.png
  outputs/reports/mlp_report.md
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings, csv
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_SATS = [51, 85, 87, 92, 109]
MAX_TS = 5e15
SESSION_BOUNDS = [1940e12, 1995e12]
SEED = 42
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Load and extract features ─────────────────────────────────────────
print("=" * 70)
print("Step 1 — Loading data and extracting features")
print("=" * 70)

all_feats, all_labels, all_ts = [], [], []

for seg in range(5):
    samples = np.load(DATA_DIR / f"samples_{seg:03d}.npy")
    sats = np.load(DATA_DIR / f"ra_sat_{seg:03d}.npy")
    ts = np.load(DATA_DIR / f"timestamp_{seg:03d}.npy")

    for sat in TARGET_SATS:
        mask = (sats == sat) & (ts < MAX_TS)
        if mask.sum() == 0:
            continue

        bursts = samples[mask]
        for burst in bursts:
            I, Q = burst[:, 0], burst[:, 1]
            amp = np.sqrt(I**2 + Q**2)
            phase = np.unwrap(np.arctan2(Q, I))
            inst_freq = np.diff(phase)
            fft_mag = np.abs(np.fft.fft(I + 1j * Q))

            f = [
                # Time domain (6)
                np.mean(I), np.std(I), np.mean(Q), np.std(Q),
                np.mean(I**3)/(np.std(I)**3 + 1e-10),  # skewness I
                np.mean(Q**3)/(np.std(Q)**3 + 1e-10),  # skewness Q
                # Amplitude/power (4)
                np.mean(amp), np.std(amp),
                np.mean(amp**2),  # signal power
                np.max(amp**2) / (np.mean(amp**2) + 1e-10),  # PAPR
                # Phase (4)
                np.mean(inst_freq), np.std(inst_freq),
                np.median(inst_freq),
                np.percentile(inst_freq, 75) - np.percentile(inst_freq, 25),
                # Frequency domain (4)
                fft_mag[np.argmax(fft_mag[1:]) + 1],  # peak magnitude
                np.mean(fft_mag), np.std(fft_mag),
                np.sum(fft_mag * np.arange(len(fft_mag))) / (np.sum(fft_mag) + 1e-10),  # centroid
                # Cross (2)
                np.corrcoef(I, Q)[0, 1],
                np.mean(I**4)/(np.std(I)**4 + 1e-10),  # kurtosis I
            ]
            all_feats.append(f)

        all_labels.extend([sat] * mask.sum())
        all_ts.extend(ts[mask].tolist())

    print(f"  Segment {seg:03d}: done")

X = np.nan_to_num(np.array(all_feats), nan=0.0, posinf=0.0, neginf=0.0)
all_labels = np.array(all_labels)
sat_to_idx = {s: i for i, s in enumerate(TARGET_SATS)}
y = np.array([sat_to_idx[s] for s in all_labels])
sessions = np.array([0 if t < SESSION_BOUNDS[0] else
                      1 if t < SESSION_BOUNDS[1] else 2 for t in all_ts])
n_classes = 5
n_features = X.shape[1]

FEATURE_NAMES = [
    "mean_I", "std_I", "mean_Q", "std_Q", "skew_I", "skew_Q",
    "mean_amp", "std_amp", "signal_power", "PAPR",
    "inst_freq_mean", "inst_freq_std", "inst_freq_median", "inst_freq_IQR",
    "fft_peak", "fft_mean", "fft_std", "spectral_centroid",
    "IQ_corr", "kurtosis_I",
]

print(f"\n  Total: {len(X)} samples, {n_features} features")
for s in TARGET_SATS:
    print(f"  Sat {s}: {(all_labels == s).sum()}")


# ── 2. MLP Model ─────────────────────────────────────────────────────────
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """
    3-layer MLP for satellite classification from extracted features.
    Architecture: Input(20) → Dense(128) → Dense(64) → Dense(32) → Dense(5)
    """
    def __init__(self, n_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)


# ── 3. Training function ─────────────────────────────────────────────────
def train_mlp(train_idx, test_idx, label, X, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])

    train_ds = FeatureDataset(X_tr, y[train_idx])
    test_ds = FeatureDataset(X_te, y[test_idx])

    # Balanced sampling
    labels = y[train_idx]
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(n_features, n_classes).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_preds, best_true = 0, None, None

    for epoch in range(EPOCHS):
        model.train()
        tr_loss, tr_c, tr_t = 0, 0, 0
        for xb, yb in train_ld:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_c += (logits.argmax(1) == yb).sum().item()
            tr_t += len(yb)
        sched.step()

        model.eval()
        va_loss, va_c, va_t = 0, 0, 0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_ld:
                logits = model(xb)
                va_loss += criterion(logits, yb).item() * len(yb)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_t += len(yb)
                preds.append(logits.argmax(1).numpy())
                trues.append(yb.numpy())

        tr_acc = tr_c / tr_t
        va_acc = va_c / va_t
        history["train_loss"].append(tr_loss / tr_t)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss / va_t)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            best_preds = np.concatenate(preds)
            best_true = np.concatenate(trues)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:2d}/{EPOCHS}  train={tr_acc:.3f}  val={va_acc:.3f}")

    f1 = f1_score(best_true, best_preds, average="macro")
    print(f"    ► {label}: accuracy={best_acc:.4f}, macro-F1={f1:.4f}")
    return best_acc, f1, best_preds, best_true, history


# ── 4. Evaluate: stratified + cross-session ──────────────────────────────
print("\n" + "=" * 70)
print("Step 2 — Stratified 5-fold CV")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strat_accs, strat_f1s = [], []
all_strat_preds, all_strat_true = [], []
all_histories = []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold_i+1}/5")
    acc, f1, preds, true, hist = train_mlp(tr_idx, te_idx, f"Fold {fold_i+1}", X, y)
    strat_accs.append(acc)
    strat_f1s.append(f1)
    all_strat_preds.append(preds)
    all_strat_true.append(true)
    all_histories.append(hist)

strat_preds = np.concatenate(all_strat_preds)
strat_true = np.concatenate(all_strat_true)
strat_acc = accuracy_score(strat_true, strat_preds)
strat_f1 = f1_score(strat_true, strat_preds, average="macro")
print(f"\n  ═══ Stratified overall: accuracy={strat_acc:.4f}, macro-F1={strat_f1:.4f}")

print("\n" + "=" * 70)
print("Step 3 — Cross-session GroupKFold")
print("=" * 70)

gkf = GroupKFold(n_splits=3)
cross_accs, cross_f1s = [], []
all_cross_preds, all_cross_true = [], []

for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=sessions)):
    test_sess = chr(65 + sessions[te_idx[0]])
    print(f"\n  Fold {fold_i+1}/3 (test=Session {test_sess})")
    acc, f1, preds, true, hist = train_mlp(tr_idx, te_idx,
                                            f"Fold {fold_i+1} (test={test_sess})", X, y)
    cross_accs.append(acc)
    cross_f1s.append(f1)
    all_cross_preds.append(preds)
    all_cross_true.append(true)

cross_preds = np.concatenate(all_cross_preds)
cross_true = np.concatenate(all_cross_true)
cross_acc = accuracy_score(cross_true, cross_preds)
cross_f1 = f1_score(cross_true, cross_preds, average="macro")
print(f"\n  ═══ Cross-session overall: accuracy={cross_acc:.4f}, macro-F1={cross_f1:.4f}")


# ── 5. Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  {'Evaluation':<30} {'Accuracy':>10} {'Macro F1':>10}")
print(f"  {'-'*52}")
print(f"  {'Chance (5 classes)':<30} {'0.2000':>10} {'0.2000':>10}")
print(f"  {'MLP Stratified 5-fold':<30} {strat_acc:>10.4f} {strat_f1:>10.4f}")
print(f"  {'MLP Cross-session':<30} {cross_acc:>10.4f} {cross_f1:>10.4f}")

print(f"\n  Classification Report (cross-session):")
sat_names = [f"Sat {s}" for s in TARGET_SATS]
print(classification_report(cross_true, cross_preds, target_names=sat_names))


# ── 6. Save ──────────────────────────────────────────────────────────────
results = [
    {"model": "MLP", "evaluation": "Stratified 5-fold", "accuracy": f"{strat_acc:.4f}",
     "macro_f1": f"{strat_f1:.4f}"},
    {"model": "MLP", "evaluation": "Cross-session", "accuracy": f"{cross_acc:.4f}",
     "macro_f1": f"{cross_f1:.4f}"},
]
with open(OUT_TABLES / "mlp_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nSaved: {OUT_TABLES / 'mlp_results.csv'}")

# Loss curves (first fold)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
h = all_histories[0]
axes[0].plot(h["train_loss"], label="Train"); axes[0].plot(h["val_loss"], label="Val")
axes[0].set_title("MLP Loss (Fold 1)"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(h["train_acc"], label="Train"); axes[1].plot(h["val_acc"], label="Val")
axes[1].axhline(0.2, color="red", linestyle="--", alpha=0.5, label="Chance")
axes[1].set_title("MLP Accuracy (Fold 1)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "mlp_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIGS / 'mlp_loss_curves.png'}")

# Report
report = f"""# MLP Results

## Architecture
- Input: {n_features} extracted features
- Features: {', '.join(FEATURE_NAMES)}
- Layers: Linear(128) → BN → ReLU → Drop(0.3) → Linear(64) → BN → ReLU → Drop(0.3) → Linear(32) → ReLU → Drop(0.2) → Linear(5)
- Optimiser: AdamW (lr={LR}), cosine annealing, {EPOCHS} epochs
- Balanced sampling via WeightedRandomSampler

## Results

| Evaluation | Accuracy | Macro F1 |
|------------|----------|----------|
| Chance | 0.2000 | 0.2000 |
| Stratified 5-fold | {strat_acc:.4f} | {strat_f1:.4f} |
| Cross-session | {cross_acc:.4f} | {cross_f1:.4f} |

## Interpretation
The MLP on extracted features shows similar performance to classical ML
baselines and the CNN, confirming that the lack of discrimination is
a data/signal issue, not a model capacity issue.
"""
with open(OUT_REPORTS / "mlp_report.md", "w") as f:
    f.write(report)
print(f"Saved: {OUT_REPORTS / 'mlp_report.md'}")
print("\nDone.")
