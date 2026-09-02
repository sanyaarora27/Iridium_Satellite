#!/usr/bin/env python3
"""Shared utilities for the raw-IQ CNN experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_SATS = [51, 85, 87, 92, 109]
N_SEGMENTS = 5
BURST_LEN = 11000
MAX_VALID_TS = 5e15
SESSION_BOUNDARIES = [1940e12, 1995e12]

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_inner_validation(
    outer_train_idx,
    labels,
    session_groups,
    evaluation_type,
    seed=42,
    validation_fraction=0.2,
):
    """Create deterministic train-only inner train/validation indices."""
    outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
    if evaluation_type == "cross-session":
        sessions = np.sort(np.unique(session_groups[outer_train_idx]))
        if len(sessions) >= 2:
            validation_session = sessions[-1]
            inner_validation = outer_train_idx[
                session_groups[outer_train_idx] == validation_session
            ]
            inner_train = outer_train_idx[
                session_groups[outer_train_idx] != validation_session
            ]
            return inner_train, inner_validation, "held-out inner session"

    inner_train, inner_validation = train_test_split(
        outer_train_idx,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels[outer_train_idx],
    )
    return inner_train, inner_validation, "stratified train-only split"

def select_best_validation_epoch(validation_losses):
    """Select the earliest epoch with the lowest validation loss."""
    if not validation_losses:
        raise ValueError("At least one validation loss is required")
    return min(range(len(validation_losses)), key=lambda epoch: validation_losses[epoch])

def assign_session(timestamp, session_boundaries=None):
    if session_boundaries is None:
        session_boundaries = SESSION_BOUNDARIES
    for i, boundary in enumerate(session_boundaries):
        if timestamp < boundary:
            return i
    return len(session_boundaries)

def load_raw_iq_data(
    target_sats=None,
    data_dir=DATA_DIR,
    n_segments=N_SEGMENTS,
    max_valid_ts=MAX_VALID_TS,
    downsample=1,
):
    if target_sats is None:
        target_sats = TARGET_SATS

    all_iq, all_labels, all_ts = [], [], []
    for seg in range(n_segments):
        samples = np.load(data_dir / f"samples_{seg:03d}.npy")
        sats = np.load(data_dir / f"ra_sat_{seg:03d}.npy")
        timestamps = np.load(data_dir / f"timestamp_{seg:03d}.npy")

        for sat in target_sats:
            mask = (sats == sat) & (timestamps < max_valid_ts)
            if mask.sum() == 0:
                continue
            iq = samples[mask]
            if downsample != 1:
                iq = iq[:, ::downsample, :]
            all_iq.append(iq)
            all_labels.append(np.full(iq.shape[0], sat))
            all_ts.append(timestamps[mask])

    if not all_iq:
        raise ValueError("No bursts loaded for the requested satellites and time window.")

    all_iq = np.concatenate(all_iq)
    all_labels = np.concatenate(all_labels)
    all_ts = np.concatenate(all_ts)

    sat_to_idx = {sat: i for i, sat in enumerate(target_sats)}
    labels_idx = np.array([sat_to_idx[s] for s in all_labels], dtype=np.int64)
    session_groups = np.array([assign_session(ts) for ts in all_ts])
    return all_iq, labels_idx, all_ts, session_groups, sat_to_idx

class IQDataset(Dataset):
    """Complex IQ bursts converted to 2-channel real tensors."""

    def __init__(self, iq, labels, burst_len=BURST_LEN, augment=False, normalise="per_burst"):
        self.iq = iq
        self.labels = labels
        self.burst_len = burst_len
        self.augment = augment
        self.normalise = normalise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = np.asarray(self.iq[idx], dtype=np.float32)
        if sample.ndim == 2 and sample.shape[-1] == 2:
            x = sample.T.copy()
        elif sample.ndim == 2:
            x = sample.copy()
        elif sample.ndim == 3 and sample.shape[-1] == 2:
            x = sample.T.copy()
        else:
            raise ValueError(f"Unexpected IQ shape: {sample.shape}")

        if x.shape[1] > self.burst_len:
            x = x[:, : self.burst_len]
        elif x.shape[1] < self.burst_len:
            pad = self.burst_len - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

        x = x.astype(np.float32)

        if self.augment:
            noise_std = np.std(x) * 0.02
            x = x + np.random.randn(*x.shape).astype(np.float32) * noise_std
            x = x * np.random.uniform(0.93, 1.07)
            shift = np.random.randint(-200, 200)
            x = np.roll(x, shift, axis=1)

        if self.normalise == "per_burst":
            for ch in range(x.shape[0]):
                mu = x[ch].mean()
                sd = x[ch].std() + 1e-8
                x[ch] = (x[ch] - mu) / sd
        elif self.normalise == "global_minmax":
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class SatCNN(nn.Module):
    """3-block 1D CNN used across the raw-IQ experiments."""

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

        self.gap = nn.AdaptiveAvgPool1d(1)

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
        return self.classifier(x)

class DualPoolCNN(nn.Module):
    """Alternative 1D-CNN with GAP + GMP for dual-evaluation experiments."""

    def __init__(self, n_classes=5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=21, stride=2, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
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
        avg = self.gap(x).squeeze(-1)
        mx = self.gmp(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)
        return self.classifier(x)

class FastCNN(nn.Module):
    """Fast, downsampled architecture used in the quick CNN check."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))

def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y_batch in loader:
        X = X.to(device)
        y_batch = y_batch.to(device)
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
def evaluate(model, loader, criterion=None, device="cpu"):
    model.eval()
    all_logits, all_y = [], []
    total_loss, correct, total = 0.0, 0, 0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    for X, y_batch in loader:
        X = X.to(device)
        y_batch = y_batch.to(device)
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
