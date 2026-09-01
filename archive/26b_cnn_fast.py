#!/usr/bin/env python3
"""
26b_cnn_fast.py – Quick CNN with downsampled IQ
================================================
This fast path keeps the same experiment but reuses the shared raw-IQ utilities.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from scripts.cnn_common import (
    DATA_DIR, FastCNN, IQDataset, TARGET_SATS, MAX_VALID_TS,
    SESSION_BOUNDARIES, load_raw_iq_data, set_seed,
)

warnings.filterwarnings("ignore")

OUT_FIGS = Path("outputs/figures")
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS = Path("outputs/reports")
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE = 4
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
DEVICE = "cpu"
SEED = 42
set_seed(SEED)

all_iq, y, all_ts, sessions, sat_to_idx = load_raw_iq_data(
    target_sats=TARGET_SATS,
    data_dir=DATA_DIR,
    n_segments=5,
    max_valid_ts=MAX_VALID_TS,
    downsample=DOWNSAMPLE,
)

n_classes = len(TARGET_SATS)
seq_len = all_iq.shape[1]
print(f"Loaded {len(all_iq)} bursts, downsampled to {seq_len} samples each")
for sat in TARGET_SATS:
    print(f"  Sat {sat}: {(y == sat_to_idx[sat]).sum()}")


def run(train_idx, test_idx, label):
    train_ds = IQDataset(all_iq[train_idx], y[train_idx], augment=True, burst_len=seq_len)
    test_ds = IQDataset(all_iq[test_idx], y[test_idx], augment=False, burst_len=seq_len)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FastCNN(n_classes=n_classes).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_preds, best_true = 0, None, None
    for epoch in range(EPOCHS):
        model.train()
        tr_c, tr_t = 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_c += (model(xb).argmax(1) == yb).sum().item()
            tr_t += len(yb)
        sched.step()
        tr_acc = tr_c / tr_t

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_ld:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        va_acc = accuracy_score(trues, preds)

        if va_acc > best_acc:
            best_acc = va_acc
            best_preds, best_true = preds.copy(), trues.copy()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:2d}/{EPOCHS}  train={tr_acc:.3f}  val={va_acc:.3f}")

    f1 = f1_score(best_true, best_preds, average="macro")
    print(f"  ► {label}: accuracy={best_acc:.4f}, macro-F1={f1:.4f}")
    return best_acc, f1, best_preds, best_true

print("\n" + "=" * 60)
print("A) STRATIFIED 80/20 SPLIT (sessions mixed)")
print("=" * 60)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(sss.split(all_iq, y))
strat_acc, strat_f1, strat_preds, strat_true = run(tr_idx, te_idx, "Stratified")

print("\n" + "=" * 60)
print("B) CROSS-SESSION (train A+B, test C)")
print("=" * 60)
tr_idx = np.where(sessions != 2)[0]
te_idx = np.where(sessions == 2)[0]
cross_acc, cross_f1, cross_preds, cross_true = run(tr_idx, te_idx, "Cross-session")

print("\n" + "=" * 60)
print("C) WITHIN SESSION A ONLY (80/20)")
print("=" * 60)
sess_a = np.where(sessions == 0)[0]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_a, te_a = next(sss2.split(all_iq[sess_a], y[sess_a]))
within_acc, within_f1, within_preds, within_true = run(sess_a[tr_a], sess_a[te_a], "Within-session")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  {'Evaluation':<35} {'Accuracy':>10} {'Macro F1':>10}")
print(f"  {'-'*57}")
print(f"  {'Chance (5 classes)':<35} {'0.2000':>10} {'0.2000':>10}")
print(f"  {'Within-session A (80/20)':<35} {within_acc:>10.4f} {within_f1:>10.4f}")
print(f"  {'Stratified 80/20 (mixed)':<35} {strat_acc:>10.4f} {strat_f1:>10.4f}")
print(f"  {'Cross-session (train AB, test C)':<35} {cross_acc:>10.4f} {cross_f1:>10.4f}")

sat_names = [f"Sat {s}" for s in TARGET_SATS]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (preds, true, title) in zip(axes, [
    (within_preds, within_true, f"Within-session A\nacc={within_acc:.4f}"),
    (strat_preds, strat_true, f"Stratified mixed\nacc={strat_acc:.4f}"),
    (cross_preds, cross_true, f"Cross-session\nacc={cross_acc:.4f}"),
]):
    cm = confusion_matrix(true, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sat_names, yticklabels=sat_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

plt.suptitle("CNN: Three Evaluation Regimes Compared", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_three_evals.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT_FIGS / 'cnn_three_evals.png'}")
print("\nDone.")
