#!/usr/bin/env python3
"""
26_cnn_dual_eval.py – 1D-CNN with dual evaluation strategy
============================================================
This script preserves the original dual-evaluation experiment while reusing the
shared raw-IQ CNN utilities from scripts/cnn_common.py.
"""

import warnings
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from scripts.cnn_common import (
        BURST_LEN, DATA_DIR, DualPoolCNN, IQDataset, MAX_VALID_TS,
        TARGET_SATS, evaluate, load_raw_iq_data, select_best_validation_epoch,
        set_seed, split_inner_validation, train_one_epoch,
    )
except ModuleNotFoundError:
    from cnn_common import (
        BURST_LEN, DATA_DIR, DualPoolCNN, IQDataset, MAX_VALID_TS,
        TARGET_SATS, evaluate, load_raw_iq_data, select_best_validation_epoch,
        set_seed, split_inner_validation, train_one_epoch,
    )

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures"
OUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
for path in [OUT_TABLES, OUT_FIGS, OUT_REPORTS]:
    path.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
set_seed(SEED)

print("=" * 70)
print("Step 1 – Loading all segments")
print("=" * 70)
all_iq, y, _, session_groups, sat_to_idx = load_raw_iq_data(
    target_sats=TARGET_SATS,
    data_dir=DATA_DIR,
    n_segments=5,
    max_valid_ts=MAX_VALID_TS,
)

n_classes = len(TARGET_SATS)
print(f"\n  Total bursts: {len(all_iq)}")
for sat in TARGET_SATS:
    print(f"  Satellite {sat}: {(y == sat_to_idx[sat]).sum()}")
print(f"\n  Session A: {(session_groups == 0).sum()} | B: {(session_groups == 1).sum()} | C: {(session_groups == 2).sum()}")

def run_fold(train_idx, test_idx, fold_name, evaluation_type, n_epochs=EPOCHS):
    inner_train_idx, inner_validation_idx, validation_method = split_inner_validation(
        train_idx, y, session_groups, evaluation_type, seed=SEED
    )
    if set(test_idx).intersection(inner_train_idx) or set(test_idx).intersection(inner_validation_idx):
        raise AssertionError("Outer test indices leaked into inner training or validation")
    if evaluation_type == "cross-session":
        test_sessions = set(session_groups[test_idx])
        if test_sessions.intersection(session_groups[inner_train_idx]) or test_sessions.intersection(session_groups[inner_validation_idx]):
            raise AssertionError("Outer test session leaked into inner training or validation")

    train_ds = IQDataset(all_iq[inner_train_idx], y[inner_train_idx], augment=True, burst_len=BURST_LEN)
    validation_ds = IQDataset(all_iq[inner_validation_idx], y[inner_validation_idx], augment=False, burst_len=BURST_LEN)
    test_ds = IQDataset(all_iq[test_idx], y[test_idx], augment=False, burst_len=BURST_LEN)

    train_labels = y[inner_train_idx]
    class_counts = np.bincount(train_labels, minlength=n_classes)
    weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = DualPoolCNN(n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    validation_losses = []
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=DEVICE)
        va_loss, va_acc, _, _, _ = evaluate(model, validation_loader, criterion=criterion, device=DEVICE)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        validation_losses.append(va_loss)
        best_epoch = select_best_validation_epoch(validation_losses)
        if best_epoch == epoch:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  val_loss={va_loss:.4f}  val_acc={va_acc:.3f}")

    selected_epoch = select_best_validation_epoch(validation_losses)
    model.load_state_dict(best_state)
    _, _, preds, y_true, probs = evaluate(model, test_loader, criterion=criterion, device=DEVICE)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    print(f"    ► {fold_name} epoch {selected_epoch + 1} test: accuracy={acc:.4f}, macro-F1={f1:.4f}")
    metadata = {
        "inner_validation_method": validation_method,
        "selected_epoch": selected_epoch + 1,
        "selection_metric": "validation_loss_min",
        "validation_score_at_selected_epoch": validation_losses[selected_epoch],
        "random_seed": SEED,
        "n_inner_train": len(inner_train_idx),
        "n_inner_validation": len(inner_validation_idx),
        "n_outer_test": len(test_idx),
    }
    return acc, f1, preds, y_true, probs, history, metadata

print("\n" + "=" * 70)
print("EVALUATION A: Stratified 5-Fold CV (sessions mixed)")
print("  This allows session leakage — test bursts share sessions with training")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strat_results, strat_all_preds, strat_all_true, strat_all_probs = [], [], [], []

for fold_i, (train_idx, test_idx) in enumerate(skf.split(all_iq, y)):
    print(f"\n  Fold {fold_i+1}/5 (train={len(train_idx)}, test={len(test_idx)})")
    acc, f1, preds, yt, probs, _, metadata = run_fold(train_idx, test_idx, f"Strat fold {fold_i+1}", "mixed")
    strat_results.append({"outer_evaluation_type": "mixed", "outer_fold": fold_i + 1, "outer_test_session": "mixed", "accuracy": acc, "macro_f1": f1, **metadata})
    strat_all_preds.append(preds)
    strat_all_true.append(yt)
    strat_all_probs.append(probs)

strat_preds = np.concatenate(strat_all_preds)
strat_true = np.concatenate(strat_all_true)
strat_probs = np.concatenate(strat_all_probs)
strat_acc = accuracy_score(strat_true, strat_preds)
strat_f1 = f1_score(strat_true, strat_preds, average="macro")
print(f"\n  ═══ Stratified CV overall: accuracy={strat_acc:.4f}, macro-F1={strat_f1:.4f}")

print("\n" + "=" * 70)
print("EVALUATION B: 3-Fold Cross-Session GroupKFold")
print("  No session leakage — test session never seen in training")
print("=" * 70)

gkf = GroupKFold(n_splits=3)
cross_results, cross_all_preds, cross_all_true, cross_all_probs = [], [], [], []

for fold_i, (train_idx, test_idx) in enumerate(gkf.split(all_iq, y, groups=session_groups)):
    test_sess = chr(65 + session_groups[test_idx][0])
    print(f"\n  Fold {fold_i+1}/3 (test=Session {test_sess}, train={len(train_idx)}, test={len(test_idx)})")
    acc, f1, preds, yt, probs, _, metadata = run_fold(train_idx, test_idx, f"Cross fold {fold_i+1}", "cross-session")
    cross_results.append({"outer_evaluation_type": "cross-session", "outer_fold": fold_i + 1, "outer_test_session": test_sess, "accuracy": acc, "macro_f1": f1, **metadata})
    cross_all_preds.append(preds)
    cross_all_true.append(yt)
    cross_all_probs.append(probs)

cross_preds = np.concatenate(cross_all_preds)
cross_true = np.concatenate(cross_all_true)
cross_probs = np.concatenate(cross_all_probs)
cross_acc = accuracy_score(cross_true, cross_preds)
cross_f1 = f1_score(cross_true, cross_preds, average="macro")
print(f"\n  ═══ Cross-session overall: accuracy={cross_acc:.4f}, macro-F1={cross_f1:.4f}")

result_rows = strat_results + cross_results
result_fields = [
    "outer_evaluation_type", "outer_fold", "outer_test_session",
    "inner_validation_method", "selected_epoch", "selection_metric",
    "validation_score_at_selected_epoch", "random_seed", "n_inner_train",
    "n_inner_validation", "n_outer_test", "accuracy", "macro_f1",
]
with open(OUT_TABLES / "cnn_dual_eval_results.csv", "w", newline="") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=result_fields)
    writer.writeheader()
    writer.writerows(result_rows)
print(f"  Saved: {OUT_TABLES / 'cnn_dual_eval_results.csv'}")

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

fig, ax = plt.subplots(figsize=(8, 5))
methods = ["Chance", "Stratified\n5-fold CV", "Cross-session\nGroupKFold"]
accs = [0.20, strat_acc, cross_acc]
f1s = [0.20, strat_f1, cross_f1]
xs = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(xs - width / 2, accs, width, label="Accuracy", color="#4C72B0")
bars2 = ax.bar(xs + width / 2, f1s, width, label="Macro F1", color="#DD8452")
ax.set_ylabel("Score")
ax.set_title("CNN Evaluation: Stratified CV vs Cross-Session")
ax.set_xticks(xs)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, max(max(accs), max(f1s)) * 1.3)
ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_dual_eval_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {OUT_FIGS / 'cnn_dual_eval_comparison.png'}")

sat_names = [f"Sat {s}" for s in TARGET_SATS]
cm_strat = confusion_matrix(strat_true, strat_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_strat, annot=True, fmt="d", cmap="Blues", xticklabels=sat_names, yticklabels=sat_names, ax=axes[0])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].set_title(f"Stratified CV\nacc={strat_acc:.4f}, F1={strat_f1:.4f}")
cm_cross = confusion_matrix(cross_true, cross_preds)
sns.heatmap(cm_cross, annot=True, fmt="d", cmap="Oranges", xticklabels=sat_names, yticklabels=sat_names, ax=axes[1])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_title(f"Cross-Session GroupKFold\nacc={cross_acc:.4f}, F1={cross_f1:.4f}")
plt.suptitle("Confusion Matrices: Session Leakage vs Honest Evaluation", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIGS / "cnn_stratified_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_FIGS / 'cnn_stratified_confusion.png'}")

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
    report += f"| Fold {r['outer_fold']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |\n"

    report += """
### Cross-session per fold
"""
for r in cross_results:
    report += (f"| Fold {r['outer_fold']} (test={r['outer_test_session']}) | "
               f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} |\n")

report += """
## Checkpoint selection provenance

The outer test fold is untouched during training and is evaluated once only
after checkpoint selection. Checkpoints are selected by minimum inner
validation loss. The CNN session groups are broader global timestamp windows,
not the 300-second timestamp-gap inferred passes used by the classical
evaluations.

### Mixed-session folds
"""
for r in strat_results:
    report += (f"| Fold {r['outer_fold']} | inner method={r['inner_validation_method']} | "
               f"selected epoch={r['selected_epoch']} | validation loss="
               f"{r['validation_score_at_selected_epoch']:.6f} | "
               f"inner train={r['n_inner_train']} | inner validation="
               f"{r['n_inner_validation']} | outer test={r['n_outer_test']} |\n")

report += """
### Cross-session folds
"""
for r in cross_results:
    report += (f"| Fold {r['outer_fold']} | test={r['outer_test_session']} | "
               f"inner method={r['inner_validation_method']} | selected epoch="
               f"{r['selected_epoch']} | validation loss="
               f"{r['validation_score_at_selected_epoch']:.6f} | "
               f"inner train={r['n_inner_train']} | inner validation="
               f"{r['n_inner_validation']} | outer test={r['n_outer_test']} |\n")

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
