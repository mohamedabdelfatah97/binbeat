"""
dataset.py
PyTorch Dataset for MIT-BIH preprocessed heartbeat segments.
Loads X_train.npy / X_test.npy and y_train.npy / y_test.npy
from data/processed/ and returns (signal, label) pairs.

Usage:
    from bin_main.dataset import MITBIHDataset, get_dataloaders
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")


class MITBIHDataset(Dataset):
    """
    PyTorch Dataset for MIT-BIH heartbeat segments.

    Args:
        split       : "train" or "test"
        processed_dir: path to data/processed/
        normalize   : if True, z-score normalize each beat independently
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: str = PROCESSED_DIR,
        normalize: bool = True,
    ):
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        # ── load arrays ───────────────────────────────────────────────────────
        X = np.load(os.path.join(processed_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(processed_dir, f"y_{split}.npy"))

        # ── normalize each beat independently (z-score) ───────────────────────
        if normalize:
            mean = X.mean(axis=1, keepdims=True)
            std  = X.std(axis=1, keepdims=True) + 1e-8  # avoid division by zero
            X    = (X - mean) / std

        # ── add channel dimension: (N, 187) → (N, 1, 187) ────────────────────
        # required by 1D-CNN and BNN (Conv1d expects [batch, channels, length])
        X = X[:, np.newaxis, :]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # ── load class info ───────────────────────────────────────────────────
        with open(os.path.join(processed_dir, "classes.json")) as f:
            self.classes_info = json.load(f)

        self.num_classes  = self.classes_info["num_classes"]
        self.int_to_symbol = self.classes_info["int_to_symbol"]

        print(f"[MITBIHDataset] {split}: {len(self.X)} samples, "
              f"{self.num_classes} classes, "
              f"shape={tuple(self.X.shape)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def class_counts(self):
        """Returns a dict of {class_index: count} for this split."""
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def class_weights(self):
        """
        Returns inverse-frequency weights for weighted loss.
        Helps handle class imbalance (N dominates heavily).
        """
        counts = self.class_counts()
        total  = sum(counts.values())
        weights = torch.zeros(self.num_classes)
        for cls, count in counts.items():
            weights[cls] = total / (self.num_classes * count)
        return weights


def get_dataloaders(
    processed_dir: str = PROCESSED_DIR,
    batch_size: int = 64,
    normalize: bool = True,
    num_workers: int = 0,
):
    """
    Returns (train_loader, test_loader, num_classes, class_weights).

    Args:
        processed_dir: path to data/processed/
        batch_size   : samples per batch
        normalize    : z-score normalize each beat
        num_workers  : dataloader workers (keep 0 on macOS to avoid issues)
    """
    train_dataset = MITBIHDataset("train", processed_dir, normalize)
    test_dataset  = MITBIHDataset("test",  processed_dir, normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_weights = train_dataset.class_weights()

    return train_loader, test_loader, train_dataset.num_classes, class_weights


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader, num_classes, class_weights = get_dataloaders()

    print(f"\nnum_classes   : {num_classes}")
    print(f"class_weights : {class_weights}")

    # peek at one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch X shape : {X_batch.shape}")
    print(f"Batch y shape : {y_batch.shape}")
    print(f"X dtype       : {X_batch.dtype}")
    print(f"y dtype       : {y_batch.dtype}")