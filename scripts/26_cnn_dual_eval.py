#!/usr/bin/env python3
"""
26_cnn_dual_eval.py – 1D-CNN with dual evaluation strategy
============================================================
Trains the same CNN architecture under two evaluation regimes:
  A) Stratified 5-fold CV (random splits, sessions mixed)
  B) 3-fold GroupKFold (cross-session, no session leakage)

The gap between A and B quantifies how much the model relies on
session/channel features vs genuine hardware fingerprints.

Outputs:
  outputs/tables/cnn_dual_eval.csv
  outputs/figures/cnn_dual_eval_comparison.png
  outputs/figures/cnn_stratified_confusion.png
  outputs/figures/cnn_stratified_roc.png
  outputs/reports/cnn_dual_eval_report.md
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data/raw")
OUT_TABLES    = Path("outputs/tables");   OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS      = Path("outputs/figures");  OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS   = Path("outputs/reports");  OUT_REPORTS.mkdir(parents=True, exist_ok=True)

TARGET_SATS   = [51, 85, 87, 92, 109]
N_SEGMENTS    = 5
BURST_LEN     = 11000
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42

SESSION_BOUNDARIES = [1940e12, 1995e12]
MAX_VALID_TS = 5e15

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1 – Loading all segments")
print("=" * 70)

all_iq, all_labels, all_ts = [], [], []
for seg in range(N_SEGMENTS):
    samples = np.load(DATA_DIR / f"samples_{seg:03d}.npy")
    sats = np.load(DATA_DIR / f"ra_sat_{seg:03d}.npy")
    ts = np.load(DATA_DIR / f"timestamp_{seg:03d}.npy")
    for sat in TARGET_SATS:
        mask = (sats == sat) & (ts < MAX_VALID_TS)
        if mask.sum() == 0:
            continue
        all_iq.append(samples[mask])
        all_labels.append(np.full(mask.sum(), sat))
        all_ts.append(ts[mask])
    print(f"  Segment {seg:03d}: loaded")

all_iq = np.concatenate(all_iq)
all_labels = np.concatenate(all_labels)
all_ts = np.concatenate(all_ts)

sat_to_idx = {s: i for i, s in enumerate(TARGET_SATS)}
idx_to_sat = {i: s for s, i in sat_to_idx.items()}
y = np.array([sat_to_idx[s] for s in all_labels])
n_classes = len(TARGET_SATS)

sessions = np.array([0 if t < SESSION_BOUNDARIES[0] else
                      1 if t < SESSION_BOUNDARIES[1] else 2 for t in all_ts])

print(f"\n  Total bursts: {len(all_iq)}")
for sat in TARGET_SATS:
    print(f"  Satellite {sat}: {(all_labels == sat).sum()}")
print(f"\n  Session A: {(sessions==0).sum()} | B: {(sessions==1).sum()} | C: {(sessions==2).sum()}")


# ── 2. Dataset ────────────────────────────────────────────────────────────
class IQDataset(Dataset):
    def __init__(self, iq, labels, augment=False):
        self.iq = iq
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iq = self.iq[idx]  # (11000, 2)

        # Truncate or pad
        if len(iq) > BURST_LEN:
            iq = iq[:BURST_LEN]
        elif len(iq) < BURST_LEN:
            iq = np.pad(iq, ((0, BURST_LEN - len(iq)), (0, 0)), mode='constant')

        x = iq.T.copy().astype(np.float32)  # (2, 11000)

        if self.augment:
            # Gaussian noise
            noise_std = np.std(x) * 0.02
            x += np.random.randn(*x.shape).astype(np.float32) * noise_std
            # Amplitude scaling
            x *= np.random.uniform(0.93, 1.07)
            # Small circular shift
            shift = np.random.randint(-200, 200)
            x = np.roll(x, shift, axis=1)

        # Per-channel zero mean, unit variance
        for ch in range(2):
            mu = x[ch].mean()
            sd = x[ch].std() + 1e-8
            x[ch] = (x[ch] - mu) / sd

        return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.long)


# ── 3. CNN Model ──────────────────────────────────────────────────────────
class SatCNN(nn.Module):
    """
    1D-CNN with less aggressive downsampling and residual-style skip
    in the deeper blocks to preserve fine-grained signal structure.
    """
    def __init__(self, n_classes=5):
        super().__init__()

        # Block 1: capture wide patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # Block 2: medium patterns
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=21, stride=2, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Block 3: fine patterns
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Block 4: deeper feature extraction
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)  # also max pool for richer summary

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 128 from GAP + 128 from GMP
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Dual pooling
        avg = self.gap(x).squeeze(-1)
        mx = self.gmp(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)  # (batch, 256)
        return self.classifier(x)


# ── 4. Training functions ────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, yb in loader:
        X, yb = X.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_y = [], []
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for X, yb in loader:
        X, yb = X.to(DEVICE), yb.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = all_logits.argmax(1).numpy()
    y_true = all_y.numpy()
    return total_loss / total, correct / total, preds, y_true, probs


def run_fold(train_idx, test_idx, fold_name, n_epochs=EPOCHS):
    """Train and evaluate one fold."""
    train_ds = IQDataset(all_iq[train_idx], y[train_idx], augment=True)
    test_ds  = IQDataset(all_iq[test_idx],  y[test_idx],  augment=False)

    # Balanced sampling
    train_labels = y[train_idx]
    class_counts = np.bincount(train_labels, minlength=n_classes)
    weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = SatCNN(n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
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

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}")

    model.load_state_dict(best_state)
    _, _, preds, y_true, probs = evaluate(model, test_loader)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"    ► {fold_name} best: accuracy={acc:.4f}, macro-F1={f1:.4f}")

    return acc, f1, preds, y_true, probs, history


# ── 5. Evaluation A: Stratified 5-fold CV ─────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION A: Stratified 5-Fold CV (sessions mixed)")
print("  This allows session leakage — test bursts share sessions with training")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strat_results = []
strat_all_preds, strat_all_true, strat_all_probs = [], [], []
strat_histories = []

for fold_i, (train_idx, test_idx) in enumerate(skf.split(all_iq, y)):
    print(f"\n  Fold {fold_i+1}/5 (train={len(train_idx)}, test={len(test_idx)})")
    acc, f1, preds, yt, probs, hist = run_fold(train_idx, test_idx, f"Strat fold {fold_i+1}")
    strat_results.append({"fold": fold_i+1, "accuracy": acc, "macro_f1": f1})
    strat_all_preds.append(preds)
    strat_all_true.append(yt)
    strat_all_probs.append(probs)
    strat_histories.append(hist)

strat_preds = np.concatenate(strat_all_preds)
strat_true = np.concatenate(strat_all_true)
strat_probs = np.concatenate(strat_all_probs)
strat_acc = accuracy_score(strat_true, strat_preds)
strat_f1 = f1_score(strat_true, strat_preds, average="macro")

print(f"\n  ═══ Stratified CV overall: accuracy={strat_acc:.4f}, macro-F1={strat_f1:.4f}")


# ── 6. Evaluation B: Cross-session GroupKFold ─────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION B: 3-Fold Cross-Session GroupKFold")
print("  No session leakage — test session never seen in training")
print("=" * 70)

gkf = GroupKFold(n_splits=3)
cross_results = []
cross_all_preds, cross_all_true, cross_all_probs = [], [], []

for fold_i, (train_idx, test_idx) in enumerate(gkf.split(all_iq, y, groups=sessions)):
    test_sess = chr(65 + sessions[test_idx][0])
    print(f"\n  Fold {fold_i+1}/3 (test=Session {test_sess}, "
          f"train={len(train_idx)}, test={len(test_idx)})")
    acc, f1, preds, yt, probs, hist = run_fold(train_idx, test_idx,
                                                 f"Cross fold {fold_i+1}")
    cross_results.append({"fold": fold_i+1, "test_session": test_sess,
                          "accuracy": acc, "macro_f1": f1})
    cross_all_preds.append(preds)
    cross_all_true.append(yt)
    cross_all_probs.append(probs)

cross_preds = np.concatenate(cross_all_preds)
cross_true = np.concatenate(cross_all_true)
cross_probs = np.concatenate(cross_all_probs)
cross_acc = accuracy_score(cross_true, cross_preds)
cross_f1 = f1_score(cross_true, cross_preds, average="macro")

print(f"\n  ═══ Cross-session overall: accuracy={cross_acc:.4f}, macro-F1={cross_f1:.4f}")


# ── 7. Comparison ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\n  {'Evaluation':<25} {'Accuracy':>10} {'Macro F1':>10}")
print(f"  {'-'*47}")
print(f"  {'Chance (5 classes)':<25} {'0.2000':>10} {'0.2000':>10}")
print(f"  {'Stratified 5-fold CV':<25} {strat_acc:>10.4f} {strat_f1:>10.4f}")
print(f"  {'Cross-session GroupKF':<25} {cross_acc:>10.4f} {cross_f1:>10.4f}")

gap = strat_acc - cross_acc
if gap > 0.05:
    print(f"\n  Gap = {gap:.1%} — model exploits session-specific features")
elif strat_acc < 0.25:
    print(f"\n  Both at chance — signal is not learnable by this CNN")
else:
    print(f"\n  Gap = {gap:.1%} — model may capture genuine hardware features")


# ── 8. Figures ────────────────────────────────────────────────────────────

# Comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
methods = ["Chance", "Stratified\n5-fold CV", "Cross-session\nGroupKFold"]
accs = [0.20, strat_acc, cross_acc]
f1s = [0.20, strat_f1, cross_f1]
x = np.arange(len(methods))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
bars2 = ax.bar(x + w/2, f1s, w, label="Macro F1", color="#DD8452")
ax.set_ylabel("Score")
ax.set_title("CNN Evaluation: Stratified CV vs Cross-Session")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, max(max(accs), max(f1s)) * 1.3)
ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1%}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_dual_eval_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT_FIGS / 'cnn_dual_eval_comparison.png'}")

# Stratified confusion matrix
sat_names = [f"Sat {s}" for s in TARGET_SATS]
cm_strat = confusion_matrix(strat_true, strat_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_strat, annot=True, fmt="d", cmap="Blues",
            xticklabels=sat_names, yticklabels=sat_names, ax=axes[0])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].set_title(f"Stratified CV\nacc={strat_acc:.4f}, F1={strat_f1:.4f}")

cm_cross = confusion_matrix(cross_true, cross_preds)
sns.heatmap(cm_cross, annot=True, fmt="d", cmap="Oranges",
            xticklabels=sat_names, yticklabels=sat_names, ax=axes[1])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_title(f"Cross-Session GroupKFold\nacc={cross_acc:.4f}, F1={cross_f1:.4f}")

plt.suptitle("Confusion Matrices: Session Leakage vs Honest Evaluation", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_stratified_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_stratified_confusion.png'}")

# Stratified ROC curves
y_bin = label_binarize(strat_true, classes=list(range(n_classes)))
fig, ax = plt.subplots(figsize=(8, 6))
for i, sat in enumerate(TARGET_SATS):
    fpr, tpr, _ = roc_curve(y_bin[:, i], strat_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"Sat {sat} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Stratified CV (sessions mixed)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_stratified_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_stratified_roc.png'}")


# ── 9. Report ─────────────────────────────────────────────────────────────
report = f"""# CNN Dual Evaluation Report

## Purpose
Compare the same CNN architecture under two evaluation regimes to
quantify how much performance depends on session/channel features
vs genuine transmitter hardware fingerprints.

## Model
- Architecture: 4-block 1D-CNN with dual pooling (GAP + GMP)
- Input: raw IQ burst (2 × {BURST_LEN}), per-channel normalised
- Blocks: Conv1d(32, k=51) → Conv1d(64, k=21) → Conv1d(128, k=11) → Conv1d(128, k=7)
- Pooling: Global Average + Global Max → concatenated (256-dim)
- Classifier: Dropout(0.5) → Dense(128) → Dense(64) → Dense({n_classes})
- Label smoothing: 0.1
- Optimiser: AdamW (lr={LR}, wd={WEIGHT_DECAY}), warmup + cosine decay
- Epochs: {EPOCHS}
- Augmentation: noise (2%), amplitude (±7%), circular shift (±200)

## Dataset
- {len(all_iq)} bursts across 5 satellites, 3 temporal sessions
- Sessions identified from timestamp clustering (not run_id)

## Results

| Evaluation | Accuracy | Macro F1 |
|------------|----------|----------|
| Chance | 0.2000 | 0.2000 |
| Stratified 5-fold CV | {strat_acc:.4f} | {strat_f1:.4f} |
| Cross-session GroupKFold | {cross_acc:.4f} | {cross_f1:.4f} |

### Stratified CV per fold
"""
for r in strat_results:
    report += f"| Fold {r['fold']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |\n"

report += f"""
### Cross-session per fold
"""
for r in cross_results:
    report += (f"| Fold {r['fold']} (test={r['test_session']}) "
               f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} |\n")

report += f"""
## Interpretation

The gap between stratified CV ({strat_acc:.1%}) and cross-session ({cross_acc:.1%})
evaluation is {gap:.1%}. """

if gap > 0.05:
    report += """A positive gap indicates the model exploits session-specific
channel features that do not generalise across recording sessions. This is
consistent with the classical ML analysis and confirms that the SatIQ dataset's
inter-satellite variation is dominated by channel conditions rather than
hardware fingerprints."""
elif strat_acc < 0.25:
    report += """Both evaluations are at chance level, confirming that neither
session-specific features nor hardware fingerprints are learnable from the
raw IQ representation by this CNN architecture."""
else:
    report += """The small gap suggests the model may capture some genuine
hardware-specific features that generalise across sessions."""

report += """

## Figures
- `cnn_dual_eval_comparison.png` — bar chart comparing both evaluations
- `cnn_stratified_confusion.png` — side-by-side confusion matrices
- `cnn_stratified_roc.png` — ROC curves for stratified CV
"""

with open(OUT_REPORTS / "cnn_dual_eval_report.md", "w") as f:
    f.write(report)
print(f"  Saved: {OUT_REPORTS / 'cnn_dual_eval_report.md'}")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
