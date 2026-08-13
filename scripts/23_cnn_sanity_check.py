"""Quick sanity check: can the CNN learn within a single session?"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

DATA_DIR = Path("data/raw")
TARGET_SATS = [51, 85, 87, 92, 109]
BURST_LEN = 11000
DEVICE = "cpu"
MAX_VALID_TS = 5e15

# Load only session A (segment 0)
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

class IQDataset(Dataset):
    def __init__(self, iq, labels, normalise="none"):
        self.iq = iq
        self.labels = labels
        self.normalise = normalise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.iq[idx].T.astype(np.float32)  # (2, 11000)
        if self.normalise == "per_burst":
            for ch in range(2):
                mu, sd = x[ch].mean(), x[ch].std() + 1e-8
                x[ch] = (x[ch] - mu) / sd
        elif self.normalise == "global_minmax":
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        # "none" = raw values
        return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.long)

# Same CNN architecture
class SatCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 64, stride=4, padding=30), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 16, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 8, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 5),
        )
    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

# 80/20 stratified split within session A
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

for norm_mode in ["none", "per_burst", "global_minmax"]:
    print(f"\n{'='*60}")
    print(f"Normalisation: {norm_mode}")
    print(f"{'='*60}")

    train_ds = IQDataset(X[train_idx], y[train_idx], normalise=norm_mode)
    test_ds  = IQDataset(X[test_idx],  y[test_idx],  normalise=norm_mode)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

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
            print(f"  Epoch {epoch+1:2d}  train_acc={tr_acc:.3f}  val_acc={vc/vt:.3f}")

print("\nIf train_acc >> 20% for any normalisation, the CNN CAN learn.")
print("If train_acc stays ~20%, the raw IQ itself lacks discrimination.")
