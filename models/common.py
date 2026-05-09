from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SamePadConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False, groups: int = 1) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        left = total // 2
        right = total - left
        return self.conv(F.pad(x, (left, right)))


class ConvBNActDrop(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float, activation: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SamePadConv1d(in_channels, out_channels, kernel_size, bias=False),
            nn.BatchNorm1d(out_channels),
            activation,
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout: float, kernel_size: int = 1) -> None:
        super().__init__()
        self.depthwise = SamePadConv1d(in_channels, in_channels, kernel_size, bias=False, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


def he_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
