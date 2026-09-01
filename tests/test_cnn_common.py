import numpy as np
import torch

from scripts.cnn_common import (
    IQDataset,
    assign_session,
    select_best_validation_epoch,
    split_inner_validation,
)


def test_assign_session_uses_timestamp_boundaries():
    assert assign_session(1e12) == 0
    assert assign_session(1950e12) == 1
    assert assign_session(2000e12) == 2


def test_iq_dataset_normalizes_and_pads():
    iq = np.zeros((2, 3, 2), dtype=np.float32)
    iq[0, :, :] = 1.0
    iq[1, :, :] = 2.0
    ds = IQDataset(iq, np.array([0, 1]), burst_len=4, augment=False)

    x, y = ds[0]
    assert x.shape == (2, 4)
    assert y.item() == 0
    assert torch.isfinite(x).all()


def test_inner_splits_are_disjoint_from_outer_test():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sessions = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    outer_train = np.array([0, 1, 2, 3, 4, 5])
    outer_test = np.array([6, 7])

    inner_train, inner_validation, _ = split_inner_validation(
        outer_train, labels, sessions, "mixed", seed=42
    )

    assert set(inner_train).isdisjoint(inner_validation)
    assert set(outer_test).isdisjoint(inner_train)
    assert set(outer_test).isdisjoint(inner_validation)


def test_cross_session_inner_validation_excludes_outer_test_session():
    labels = np.array([0, 1, 0, 1, 0, 1])
    sessions = np.array([0, 0, 1, 1, 2, 2])
    outer_train = np.array([0, 1, 2, 3])
    outer_test = np.array([4, 5])

    inner_train, inner_validation, method = split_inner_validation(
        outer_train, labels, sessions, "cross-session", seed=42
    )

    assert method == "held-out inner session"
    assert set(sessions[outer_test]).isdisjoint(sessions[inner_train])
    assert set(sessions[outer_test]).isdisjoint(sessions[inner_validation])
    assert set(inner_train).isdisjoint(inner_validation)


def test_checkpoint_selection_uses_validation_losses_only():
    validation_losses = [0.9, 0.6, 0.7]

    assert select_best_validation_epoch(validation_losses) == 1


def test_augmentation_is_enabled_only_for_training_dataset():
    iq = np.ones((1, 3, 2), dtype=np.float32)
    labels = np.array([0])

    train_dataset = IQDataset(iq, labels, augment=True, burst_len=3)
    validation_dataset = IQDataset(iq, labels, augment=False, burst_len=3)
    test_dataset = IQDataset(iq, labels, augment=False, burst_len=3)

    assert train_dataset.augment is True
    assert validation_dataset.augment is False
    assert test_dataset.augment is False
