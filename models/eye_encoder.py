from __future__ import annotations

import torch
from torch import nn

from .common import SamePadConv1d


class EyeEncoder(nn.Module):


    def __init__(self, in_channels: int = 6, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = 32
        self.conv1 = SamePadConv1d(in_channels, 16, kernel_size=8, bias=True)
        self.conv2 = SamePadConv1d(16, 32, kernel_size=8, bias=True)
        self.act = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected EM [B,{self.in_channels},T], got {tuple(x.shape)}")
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        return self.pool(x)
