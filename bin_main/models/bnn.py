"""
bnn.py
Model 3 — 1D Binary Neural Network (Brevitas)
Same structure as CNN1D but with binarized weights and activations.
First and last layers stay float32 (standard BNN practice).

Architecture:
    Input: (batch, 1, 187)
    Conv1d(1→32, k=5)          float32  — first layer always float
    QuantConv1d(32→64, k=5)    1-bit
    QuantConv1d(64→128, k=3)   1-bit
    AdaptiveAvgPool1d(1)
    Flatten                    → (batch, 128)
    QuantLinear(128→64)        1-bit
    Linear(64→num_classes)     float32  — last layer always float
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat

# 1-bit quantization configs
class BinaryAct(Int8ActPerTensorFloat):
    bit_width = 1

class BinaryWeight(Int8WeightPerTensorFloat):
    bit_width = 1
class BNN1D(nn.Module):
    """
    1D Binary Neural Network for ECG heartbeat classification.

    Args:
        num_classes : number of output classes (default 10)
        dropout     : dropout probability (default 0.4)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # first layer — float32 (preserves input amplitude information)
        self.first_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # binary block 1
        self.bin_conv1 = nn.Sequential(
            qnn.QuantConv1d(
                32, 64,
                kernel_size=5,
                padding=2,
                weight_quant=BinaryWeight,
                input_quant=BinaryAct,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # binary block 2
        self.bin_conv2 = nn.Sequential(
            qnn.QuantConv1d(
                64, 128,
                kernel_size=3,
                padding=1,
                weight_quant=BinaryWeight,
                input_quant=BinaryAct,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # binary fully connected
        self.bin_fc = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(
                128, 64,
                bias=True,
                weight_quant=BinaryWeight,
                input_quant=BinaryAct,
                return_quant_tensor=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # last layer — float32 (preserves logit resolution)
        self.last_fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.bin_conv1(x)
        x = self.bin_conv2(x)
        x = self.bin_fc(x)
        return self.last_fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    # binary layers: 1 bit per param, float layers: 32 bits per param
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, (qnn.QuantConv1d, qnn.QuantLinear)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_bits += params * 1
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