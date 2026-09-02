#!/usr/bin/env python3
"""
22_cnn_raw_iq.py  –  1D-CNN on raw IQ bursts for satellite RF fingerprinting
=============================================================================
Loads all SatIQ segments, assigns temporal session groups, trains a 1D-CNN
with 3-fold cross-session GroupKFold, and reports authentication metrics.

Outputs:
  outputs/tables/cnn_results.csv
  outputs/figures/cnn_loss_curves.png
  outputs/figures/cnn_confusion_matrix.png
  outputs/figures/cnn_roc_curves.png
  outputs/reports/cnn_report.md
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_SATS   = [51, 85, 87, 92, 109]
N_SEGMENTS    = 5
BURST_LEN     = 11000          # samples per burst (2-channel: I, Q)
BATCH_SIZE    = 32
EPOCHS        = 60
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42

# Session boundaries (from timestamp analysis)
# Session A: ~1918e15 – 1937e15  (Seg 0)
# Session B: ~1941e15 – 1979e15  (Seg 1,2,3)
# Session C: ~1998e15 – 2018e15  (Seg 4)
SESSION_BOUNDARIES = [1940e12, 1995e12]   # split points between sessions

# Corrupt timestamp threshold (Sat 85 seg 0 has a value ~7.5e16)
MAX_VALID_TIMESTAMP = 5e15

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("Step 1 – Loading all segments")
print("=" * 70)

all_iq      = []
all_labels  = []
all_ts      = []

for seg in range(N_SEGMENTS):
    samples = np.load(DATA_DIR / f"samples_{seg:03d}.npy")
    sats    = np.load(DATA_DIR / f"ra_sat_{seg:03d}.npy")
    ts      = np.load(DATA_DIR / f"timestamp_{seg:03d}.npy")

    for sat in TARGET_SATS:
        mask = (sats == sat) & (ts < MAX_VALID_TIMESTAMP)
        if mask.sum() == 0:
            continue

        iq_bursts = samples[mask]
        sat_ts    = ts[mask]

        all_iq.append(iq_bursts)
        all_labels.append(np.full(mask.sum(), sat))
        all_ts.append(sat_ts)

    print(f"  Segment {seg:03d}: loaded")

all_iq     = np.concatenate(all_iq)
all_labels = np.concatenate(all_labels)
all_ts     = np.concatenate(all_ts)

print(f"\n  Total bursts loaded: {len(all_iq)}")
print(f"  IQ shape per burst: {all_iq[0].shape}")
for sat in TARGET_SATS:
    print(f"  Satellite {sat}: {(all_labels == sat).sum()} bursts")

print("\n" + "=" * 70)
print("Step 2 – Assigning session groups from timestamps")
print("=" * 70)

def assign_session(timestamp):
    """Assign session 0, 1, or 2 based on timestamp boundaries."""
    for i, boundary in enumerate(SESSION_BOUNDARIES):
        if timestamp < boundary:
            return i
    return len(SESSION_BOUNDARIES)

session_groups = np.array([assign_session(t) for t in all_ts])

for sess in range(3):
    mask = session_groups == sess
    print(f"  Session {chr(65 + sess)}: {mask.sum()} bursts")
    for sat in TARGET_SATS:
        n = ((all_labels == sat) & mask).sum()
        if n > 0:
            print(f"    Sat {sat}: {n}")

sat_to_idx = {sat: i for i, sat in enumerate(TARGET_SATS)}
idx_to_sat = {i: sat for sat, i in sat_to_idx.items()}
y = np.array([sat_to_idx[s] for s in all_labels])
n_classes = len(TARGET_SATS)

class IQDataset(Dataset):
    """
    Converts complex IQ into 2-channel real tensor: (2, BURST_LEN).
    Optional augmentation: Gaussian noise, amplitude scaling, circular shift.
    """
    def __init__(self, iq_complex, labels, augment=False):
        self.iq = iq_complex
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iq = self.iq[idx].copy()

        # Truncate or pad to BURST_LEN
        if len(iq) > BURST_LEN:
            iq = iq[:BURST_LEN]
        elif len(iq) < BURST_LEN:
            iq = np.pad(iq, ((0, BURST_LEN - len(iq)), (0, 0)), mode='constant')

        # Transpose to (2, BURST_LEN) — channels first for Conv1d
        x = iq.T.astype(np.float32)

        if self.augment:
            noise_std = np.std(x) * 0.03
            x += np.random.randn(*x.shape).astype(np.float32) * noise_std
            scale = np.random.uniform(0.9, 1.1)
            x *= scale
            shift = np.random.randint(-500, 500)
            x = np.roll(x, shift, axis=1)

        # Per-channel normalisation
        for ch in range(2):
            mu = x[ch].mean()
            sd = x[ch].std() + 1e-8
            x[ch] = (x[ch] - mu) / sd

        return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.long)

class SatCNN(nn.Module):
    """
    3-block 1D-CNN for RF fingerprinting on raw IQ bursts.

    Block 1: Wide kernels → low-level hardware patterns (oscillator quirks,
             DAC nonlinearities)
    Block 2: Medium kernels → mid-level pattern combinations
    Block 3: Narrow kernels → high-level fingerprint features
    GlobalAvgPool → Dropout → Dense → Softmax
    """
    def __init__(self, n_classes=5, in_channels=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)   # global average pooling

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y_batch in loader:
        X, y_batch = X.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_y = [], []
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for X, y_batch in loader:
        X, y_batch = X.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
        all_logits.append(logits.cpu())
        all_y.append(y_batch.cpu())
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = all_logits.argmax(1).numpy()
    y_true = all_y.numpy()
    return total_loss / total, correct / total, preds, y_true, probs

print("\n" + "=" * 70)
print("Step 3 – 3-Fold Cross-Session GroupKFold Training")
print("=" * 70)

gkf = GroupKFold(n_splits=3)

fold_results = []
all_fold_preds = []
all_fold_true  = []
all_fold_probs = []
fold_histories = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(all_iq, y, groups=session_groups)):
    test_sessions = np.unique(session_groups[test_idx])
    train_sessions = np.unique(session_groups[train_idx])
    print(f"\n{'─' * 60}")
    print(f"Fold {fold + 1}: train on sessions {[chr(65+s) for s in train_sessions]}, "
          f"test on session {[chr(65+s) for s in test_sessions]}")
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

    # Datasets
    train_ds = IQDataset(all_iq[train_idx], y[train_idx], augment=True)
    test_ds  = IQDataset(all_iq[test_idx],  y[test_idx],  augment=False)

    # Balanced sampling for training
    train_labels = y[train_idx]
    class_counts = np.bincount(train_labels, minlength=n_classes)
    weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = SatCNN(n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, _, _, _ = evaluate(model, test_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}")

    # Load best model and get final predictions
    model.load_state_dict(best_state)
    _, _, preds, y_true, probs = evaluate(model, test_loader)

    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, average="macro")

    print(f"\n  ► Fold {fold+1} best: accuracy={acc:.4f}, macro-F1={f1:.4f}")

    fold_results.append({"fold": fold+1, "accuracy": acc, "macro_f1": f1,
                         "test_session": chr(65 + test_sessions[0]),
                         "n_train": len(train_idx), "n_test": len(test_idx)})
    all_fold_preds.append(preds)
    all_fold_true.append(y_true)
    all_fold_probs.append(probs)
    fold_histories.append(history)

print("\n" + "=" * 70)
print("Step 4 – Aggregate Results")
print("=" * 70)

all_preds = np.concatenate(all_fold_preds)
all_true  = np.concatenate(all_fold_true)
all_probs = np.concatenate(all_fold_probs)

overall_acc = accuracy_score(all_true, all_preds)
overall_f1  = f1_score(all_true, all_preds, average="macro")

print(f"\n  Overall cross-session accuracy: {overall_acc:.4f}")
print(f"  Overall cross-session macro-F1: {overall_f1:.4f}")
print(f"\n  Per-fold breakdown:")
for r in fold_results:
    print(f"    Fold {r['fold']} (test={r['test_session']}): "
          f"acc={r['accuracy']:.4f}, F1={r['macro_f1']:.4f}")

# Classification report
sat_names = [f"Sat {s}" for s in TARGET_SATS]
print(f"\n  Classification Report:")
print(classification_report(all_true, all_preds, target_names=sat_names))

# Authentication metrics
cm = confusion_matrix(all_true, all_preds)
print("  Confusion Matrix:")
print(cm)

# Per-class detection probability (Pd) and false alarm probability (Pfa)
print(f"\n  Authentication Metrics:")
print(f"  {'Satellite':>12}  {'Pd (recall)':>12}  {'Pfa':>12}  {'FAR':>12}")
for i, sat in enumerate(TARGET_SATS):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    pd  = tp / (tp + fn) if (tp + fn) > 0 else 0
    pfa = fp / (fp + tn) if (fp + tn) > 0 else 0
    far = fp / (fp + tp) if (fp + tp) > 0 else 0
    print(f"  Sat {sat:>8}  {pd:>12.4f}  {pfa:>12.4f}  {far:>12.4f}")

import csv
with open(OUT_TABLES / "cnn_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fold", "test_session", "n_train", "n_test",
                                       "accuracy", "macro_f1"])
    w.writeheader()
    w.writerows(fold_results)
    w.writerow({"fold": "mean", "test_session": "all",
                "n_train": "", "n_test": "",
                "accuracy": f"{overall_acc:.4f}", "macro_f1": f"{overall_f1:.4f}"})
print(f"\n  Saved: {OUT_TABLES / 'cnn_results.csv'}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for fold_i, hist in enumerate(fold_histories):
    # Loss
    axes[0, fold_i].plot(hist["train_loss"], label="Train", alpha=0.8)
    axes[0, fold_i].plot(hist["val_loss"],   label="Val",   alpha=0.8)
    axes[0, fold_i].set_title(f"Fold {fold_i+1} – Loss")
    axes[0, fold_i].set_xlabel("Epoch")
    axes[0, fold_i].set_ylabel("Loss")
    axes[0, fold_i].legend()
    axes[0, fold_i].grid(True, alpha=0.3)

    # Accuracy
    axes[1, fold_i].plot(hist["train_acc"], label="Train", alpha=0.8)
    axes[1, fold_i].plot(hist["val_acc"],   label="Val",   alpha=0.8)
    axes[1, fold_i].axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Chance")
    axes[1, fold_i].set_title(f"Fold {fold_i+1} – Accuracy")
    axes[1, fold_i].set_xlabel("Epoch")
    axes[1, fold_i].set_ylabel("Accuracy")
    axes[1, fold_i].legend()
    axes[1, fold_i].grid(True, alpha=0.3)

plt.suptitle(f"1D-CNN Cross-Session Training\n"
             f"Overall: acc={overall_acc:.4f}, macro-F1={overall_f1:.4f}", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_loss_curves.png'}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sat_names, yticklabels=sat_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"CNN Cross-Session Confusion Matrix\n"
             f"acc={overall_acc:.4f}, macro-F1={overall_f1:.4f}")
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_confusion_matrix.png'}")

y_bin = label_binarize(all_true, classes=list(range(n_classes)))

fig, ax = plt.subplots(figsize=(8, 6))
for i, sat in enumerate(TARGET_SATS):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"Sat {sat} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("CNN Per-Satellite ROC Curves (Cross-Session)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_roc_curves.png'}")

report = f"""# CNN Raw-IQ RF Fingerprinting Results

## Model
- Architecture: 3-block 1D-CNN (Conv1D → BatchNorm → ReLU → Pool) × 3, GAP, Dense
- Input: raw IQ burst (2 × {BURST_LEN}), per-channel normalised
- Block 1: 32 filters, kernel=64, stride=4, MaxPool(4)
- Block 2: 64 filters, kernel=16, stride=2, MaxPool(2)
- Block 3: 128 filters, kernel=8, GlobalAvgPool
- Classifier: Dropout(0.5) → Dense(64) → Dropout(0.3) → Dense({n_classes})
- Optimiser: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})
- Scheduler: CosineAnnealing over {EPOCHS} epochs
- Augmentation: Gaussian noise (3%), amplitude scaling (±10%), circular shift (±500)

## Dataset
- Source: SatIQ/Zenodo, segments 000–004
- Satellites: {TARGET_SATS}
- Total bursts: {len(all_iq)} (after filtering corrupt timestamps)
- 3 temporal sessions identified from timestamp clustering
- Evaluation: 3-fold GroupKFold by session (cross-session generalisation)

## Results

| Fold | Test Session | Train N | Test N | Accuracy | Macro F1 |
|------|-------------|---------|--------|----------|----------|
"""

for r in fold_results:
    report += (f"| {r['fold']} | {r['test_session']} | {r['n_train']} | {r['n_test']} "
               f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} |\n")
report += f"| **Mean** | **All** | | | **{overall_acc:.4f}** | **{overall_f1:.4f}** |\n"

report += f"""
## Authentication Metrics

| Satellite | Pd (recall) | Pfa | FAR |
|-----------|-------------|-----|-----|
"""
for i, sat in enumerate(TARGET_SATS):
    tp = cm[i, i]; fn = cm[i,:].sum()-tp; fp = cm[:,i].sum()-tp; tn = cm.sum()-tp-fn-fp
    pd = tp/(tp+fn) if (tp+fn)>0 else 0
    pfa = fp/(fp+tn) if (fp+tn)>0 else 0
    far = fp/(fp+tp) if (fp+tp)>0 else 0
    report += f"| Sat {sat} | {pd:.4f} | {pfa:.4f} | {far:.4f} |\n"

report += """
## Interpretation

Cross-session evaluation tests whether the CNN learns transmitter-specific
hardware fingerprints rather than session/channel-specific features. If accuracy
is well above chance (20%) but below within-session performance, it suggests the
model captures some genuine hardware signal but channel variation remains a
significant confound — consistent with prior CrossRF (2025) findings.

## Figures
- `cnn_loss_curves.png` – training/validation loss and accuracy per fold
- `cnn_confusion_matrix.png` – aggregate confusion matrix
- `cnn_roc_curves.png` – per-satellite ROC curves
"""

with open(OUT_REPORTS / "cnn_report.md", "w") as f:
    f.write(report)
print(f"  Saved: {OUT_REPORTS / 'cnn_report.md'}")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
