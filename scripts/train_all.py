"""
train_all.py
Trains all three binbeat models sequentially and saves
training histories to results/.

Usage:
    python3 scripts/train_all.py
"""

import os
import sys

# make bin_main importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from bin_main.dataset import get_dataloaders
from bin_main.models.mlp import MLP
from bin_main.models.cnn import CNN1D
from bin_main.models.bnn import BNN1D
from bin_main.train import train, save_history

# config
NUM_EPOCHS     = 100
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
RESULTS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # load data once, share across all models
    print("Loading dataset...")
    train_loader, test_loader, num_classes, class_weights = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    print(f"num_classes: {num_classes}\n")

    # define all three models
    models = {
        "mlp": MLP(num_classes=num_classes),
        "cnn": CNN1D(num_classes=num_classes),
        "bnn": BNN1D(num_classes=num_classes),
    }

    all_histories = {}

    for model_name, model in models.items():
        history = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=model_name,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            class_weights=class_weights,
            checkpoint_dir=CHECKPOINT_DIR,
        )

        all_histories[model_name] = history

        # save individual history
        save_history(
            history,
            os.path.join(RESULTS_DIR, f"{model_name}_history.json")
        )
        print(f"History saved: results/{model_name}_history.json\n")

    # save combined summary
    summary = {
        name: {
            "best_test_acc":  max(h["test_acc"]),
            "best_train_acc": max(h["train_acc"]),
            "total_time_s":   sum(h["epoch_time"]),
        }
        for name, h in all_histories.items()
    }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll models trained.")
    print("\nFinal Summary:")
    print(f"{'Model':<8} {'Best Test Acc':<18} {'Total Time'}")
    print("-" * 40)
    for name, s in summary.items():
        print(f"{name:<8} {s['best_test_acc']:<18.4f} {s['total_time_s']:.1f}s")

if __name__ == "__main__":
    main()