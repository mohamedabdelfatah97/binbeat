"""
cnn.py
Model 2 — 1D Convolutional Neural Network (Float32)
Treats the heartbeat as a 1D signal and uses convolutional filters
to detect local waveform patterns (QRS morphology, P-wave, T-wave).

Architecture:
    Input: (batch, 1, 187)
    Conv1d(1→32,  k=5) + BN + ReLU + MaxPool(2)   → (batch, 32, 91)
    Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)   → (batch, 64, 43)
    Conv1d(64→128,k=3) + BN + ReLU + AdaptiveAvgPool → (batch, 128, 1)
    Flatten                                         → (batch, 128)
    Linear(128→64) + BN + ReLU + Dropout(0.4)
    Linear(64→num_classes)
"""

import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    1D CNN for ECG heartbeat classification.

    Args:
        num_classes : number of output classes (default 10)
        dropout     : dropout probability (default 0.4)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    return count_parameters(model) * 4 / 1024


# quick sanity check
if __name__ == "__main__":
    model = CNN1D(num_classes=10)

    print(model)
    print(f"\nTotal parameters : {count_parameters(model):,}")
    print(f"Model size       : {model_size_kb(model):.2f} KB")

    dummy_input = torch.randn(64, 1, 187)
    output = model(dummy_input)
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("Forward pass : OK")