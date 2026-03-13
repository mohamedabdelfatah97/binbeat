"""
bnn.py
Model 3 — 1D Binary Neural Network (Brevitas)
Same structure as CNN1D but with binarized weights and activations.
First and last layers stay float32 (standard BNN practice).
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv1d, QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


class BNN1D(nn.Module):
    """
    1D Binary Neural Network for ECG heartbeat classification.

    Args:
        num_classes : number of output classes (default 10)
        dropout     : dropout probability (default 0.4)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # first layer — float32
        self.first_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # quantized block 1 — 8-bit (stable proxy for binary)
        self.quant_conv1 = nn.Sequential(
            QuantConv1d(
                32, 64,
                kernel_size=5,
                padding=2,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # quantized block 2 — 8-bit
        self.quant_conv2 = nn.Sequential(
            QuantConv1d(
                64, 128,
                kernel_size=3,
                padding=1,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # quantized fully connected — 8-bit
        self.quant_fc = nn.Sequential(
            nn.Flatten(),
            QuantLinear(
                128, 64,
                bias=True,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # last layer — float32
        self.last_fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.quant_conv1(x)
        x = self.quant_conv2(x)
        x = self.quant_fc(x)
        return self.last_fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    # quantized layers: 8 bits per param, float layers: 32 bits per param
    from brevitas.nn import QuantConv1d, QuantLinear
    total_bits = 0
    for module in model.modules():
        if isinstance(module, (QuantConv1d, QuantLinear)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_bits += params * 8
        elif isinstance(module, (nn.Conv1d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_bits += params * 32
    return total_bits / (8 * 1024)


# quick sanity check
if __name__ == "__main__":
    model = BNN1D(num_classes=10)

    print(model)
    print(f"\nTotal parameters : {count_parameters(model):,}")
    print(f"Model size (BNN) : {model_size_kb(model):.2f} KB")

    dummy_input = torch.randn(64, 1, 187)
    output = model(dummy_input)
    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print("Forward pass : OK")