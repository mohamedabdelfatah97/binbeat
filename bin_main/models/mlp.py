"""
mlp.py
Model 1 — Multilayer Perceptron (Float32 baseline)
Flattens the 187-sample heartbeat window and passes it through
fully connected layers. No temporal awareness whatsoever.

Architecture:
    Input: (batch, 1, 187) → flatten → (batch, 187)
    Linear(187 → 256) + BN + ReLU + Dropout(0.3)
    Linear(256 → 128) + BN + ReLU + Dropout(0.3)
    Linear(128 → 64)  + BN + ReLU
    Linear(64  → num_classes)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully connected baseline for ECG heartbeat classification.

    Args:
        num_classes : number of output classes (default 10)
        dropout     : dropout probability (default 0.3)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # block 1
            nn.Linear(187, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # output
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 187)
        x = self.flatten(x)          # → (batch, 187)
        return self.classifier(x)    # → (batch, num_classes)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    """Returns approximate model size in KB (float32 = 4 bytes per param)."""
    return count_parameters(model) * 4 / 1024


# quick sanity check
if __name__ == "__main__":
    model = MLP(num_classes=10)

    # print architecture
    print(model)
    print(f"\nTotal parameters : {count_parameters(model):,}")
    print(f"Model size       : {model_size_kb(model):.2f} KB")

    # test forward pass
    dummy_input = torch.randn(64, 1, 187)  # one batch
    output = model(dummy_input)
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("Forward pass : OK")