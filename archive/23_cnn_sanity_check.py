"""Quick sanity check: can the CNN learn within a single session?"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from scripts.cnn_common import DATA_DIR, IQDataset, SatCNN, TARGET_SATS, MAX_VALID_TS, set_seed

DEVICE = "cpu"
SEED = 42
set_seed(SEED)

print("Loading session A (segment 0)...")
samples = np.load(DATA_DIR / "samples_000.npy")
sats = np.load(DATA_DIR / "ra_sat_000.npy")
ts = np.load(DATA_DIR / "timestamp_000.npy")

mask = np.isin(sats, TARGET_SATS) & (ts < MAX_VALID_TS)
X = samples[mask]
y_sat = sats[mask]
sat_to_idx = {s: i for i, s in enumerate(TARGET_SATS)}
y = np.array([sat_to_idx[s] for s in y_sat])

print(f"Loaded {len(X)} bursts from session A")
for s in TARGET_SATS:
    print(f"  Sat {s}: {(y_sat == s).sum()}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

for norm_mode in ["none", "per_burst", "global_minmax"]:
    print(f"\n{'=' * 60}")
    print(f"Normalisation: {norm_mode}")
    print(f"{'=' * 60}")

    train_ds = IQDataset(X[train_idx], y[train_idx], normalise=norm_mode)
    test_ds = IQDataset(X[test_idx], y[test_idx], normalise=norm_mode)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = SatCNN().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
        tr_acc = correct / total

        if (epoch + 1) % 5 == 0:
            model.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    vc += (model(xb.to(DEVICE)).argmax(1) == yb.to(DEVICE)).sum().item()
                    vt += len(yb)
            print(f"  Epoch {epoch + 1:2d}  train_acc={tr_acc:.3f}  val_acc={vc / vt:.3f}")

print("\nIf train_acc >> 20% for any normalisation, the CNN CAN learn.")
print("If train_acc stays ~20%, the raw IQ itself lacks discrimination.")
