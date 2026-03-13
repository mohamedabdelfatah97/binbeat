"""
evaluate.py
Loads trained checkpoints and computes full evaluation metrics:
- Overall accuracy
- Per-class F1 score
- Confusion matrix
- Model size in KB
- Inference time in ms per sample
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin_main.dataset import get_dataloaders
from bin_main.models.mlp import MLP, model_size_kb as mlp_size
from bin_main.models.cnn import CNN1D, model_size_kb as cnn_size
from bin_main.models.bnn import BNN1D, model_size_kb as bnn_size

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "results", "checkpoints")
RESULTS_DIR    = os.path.join(REPO_ROOT, "results")


def measure_inference_time(model, loader, device, n_batches=50):
    """Measures average inference time in ms per sample."""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (X_batch, _) in enumerate(loader):
            if i >= n_batches:
                break
            X_batch = X_batch.to(device)
            t0 = time.perf_counter()
            _ = model(X_batch)
            t1 = time.perf_counter()
            batch_size = X_batch.shape[0]
            times.append((t1 - t0) / batch_size * 1000)  # ms per sample

    return float(np.mean(times))


@torch.no_grad()
def get_predictions(model, loader, device):
    """Returns all predictions and true labels for a dataloader."""
    model.eval()
    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_model(model, model_name, size_kb, test_loader, device, int_to_symbol):
    """Full evaluation of one model. Returns metrics dict."""
    print(f"\nEvaluating {model_name.upper()}...")

    preds, labels = get_predictions(model, test_loader, device)

    acc         = (preds == labels).mean()
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    f1_macro    = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    cm          = confusion_matrix(labels, preds)
    infer_time  = measure_inference_time(model, test_loader, device)

    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 macro        : {f1_macro:.4f}")
    print(f"  F1 weighted     : {f1_weighted:.4f}")
    print(f"  Model size      : {size_kb:.2f} KB")
    print(f"  Inference time  : {infer_time:.4f} ms/sample")
    print(f"\n  Per-class F1:")
    for i, f1 in enumerate(f1_per_class):
        symbol = int_to_symbol.get(str(i), str(i))
        print(f"    {symbol:<6} : {f1:.4f}")

    return {
        "model":        model_name,
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "f1_per_class": {
            int_to_symbol.get(str(i), str(i)): float(f)
            for i, f in enumerate(f1_per_class)
        },
        "confusion_matrix": cm.tolist(),
        "model_size_kb": size_kb,
        "inference_ms":  infer_time,
    }


def run_evaluation():
    device = torch.device("cpu")

    print("Loading dataset...")
    _, test_loader, num_classes, _ = get_dataloaders(batch_size=64)

    # load class mapping
    with open(os.path.join(REPO_ROOT, "data", "processed", "classes.json")) as f:
        classes_info = json.load(f)
    int_to_symbol = classes_info["int_to_symbol"]

    # define models
    models = {
        "mlp": (MLP(num_classes=num_classes),  mlp_size(MLP(num_classes=num_classes))),
        "cnn": (CNN1D(num_classes=num_classes), cnn_size(CNN1D(num_classes=num_classes))),
        "bnn": (BNN1D(num_classes=num_classes), bnn_size(BNN1D(num_classes=num_classes))),
    }

    all_metrics = {}

    for model_name, (model, size_kb) in models.items():
        checkpoint = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
        if not os.path.exists(checkpoint):
            print(f"WARNING: no checkpoint found for {model_name}, skipping")
            continue

        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)

        metrics = evaluate_model(
            model, model_name, size_kb, test_loader, device, int_to_symbol
        )
        all_metrics[model_name] = metrics

    # save metrics
    out_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # print final comparison table
    print("\nFinal Comparison:")
    print(f"{'Model':<8} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<14} {'Size KB':<12} {'ms/sample'}")
    print("-" * 70)
    for name, m in all_metrics.items():
        print(
            f"{name:<8} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} "
            f"{m['f1_weighted']:<14.4f} {m['model_size_kb']:<12.2f} {m['inference_ms']:.4f}"
        )

    return all_metrics


if __name__ == "__main__":
    run_evaluation()