"""
train.py
Shared training loop for all three binbeat models.
Optimized for CPU training on MacBook Air M4.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in tqdm(loader, leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates model on a dataloader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    num_epochs: int = 30,
    lr: float = 1e-3,
    class_weights: torch.Tensor = None,
    checkpoint_dir: str = "results/checkpoints",
) -> dict:
    """
    Full training run for one model.

    Args:
        model         : the model to train
        train_loader  : training DataLoader
        test_loader   : test DataLoader
        model_name    : used for checkpoint filename (mlp / cnn / bnn)
        num_epochs    : number of training epochs
        lr            : learning rate
        class_weights : inverse frequency weights for imbalanced classes
        checkpoint_dir: where to save best model weights

    Returns:
        history dict with train/test loss and accuracy per epoch
    """
    # M4 Mac — use CPU, MPS has compatibility issues with Brevitas
    device = torch.device("cpu")
    model  = model.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # weighted cross entropy to handle class imbalance
    if class_weights is not None:
        class_weights = torch.clamp(class_weights, max=10.0)
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    history = {
        "model":      model_name,
        "train_loss": [],
        "train_acc":  [],
        "test_loss":  [],
        "test_acc":   [],
        "epoch_time": [],
    }

    best_test_acc  = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    print(f"\nTraining {model_name.upper()} for {num_epochs} epochs on {device}")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Test Loss':<14} {'Test Acc':<12} {'Time'}")
    print("-" * 75)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        elapsed = time.time() - t0
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(elapsed)

        print(
            f"{epoch:<8} {train_loss:<14.4f} {train_acc:<14.4f} "
            f"{test_loss:<14.4f} {test_acc:<12.4f} {elapsed:.1f}s"
        )

        # save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nBest test accuracy : {best_test_acc:.4f}")
    print(f"Checkpoint saved   : {checkpoint_path}")

    return history


def save_history(history: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# quick sanity check
if __name__ == "__main__":
    from bin_main.dataset import get_dataloaders
    from bin_main.models.mlp import MLP

    train_loader, test_loader, num_classes, class_weights = get_dataloaders(batch_size=64)

    model = MLP(num_classes=num_classes)

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        model_name="mlp",
        num_epochs=2,  # just 2 epochs to verify the loop works
        lr=1e-3,
        class_weights=class_weights,
    )

    print("\nHistory keys:", list(history.keys()))
    print("Train acc per epoch:", history["train_acc"])