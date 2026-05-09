from __future__ import annotations

import torch
from torch import nn

from .common import ConvBNActDrop, DepthwiseSeparableConv1d


class EEGEncoder(nn.Module):

    def __init__(self, in_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = 32
        self.ms1 = nn.ModuleList([ConvBNActDrop(in_channels, 8, k, dropout, nn.ELU()) for k in (64, 32, 16, 8)])
        self.ds = nn.ModuleList([DepthwiseSeparableConv1d(8, 16, dropout, kernel_size=1) for _ in range(4)])
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.ms2 = nn.ModuleList([ConvBNActDrop(64, 8, k, dropout, nn.ELU()) for k in (32, 16, 8, 4)])
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected EEG [B,{self.in_channels},T], got {tuple(x.shape)}")
        branches = [conv(x) for conv in self.ms1]
        branches = [ds(b) for ds, b in zip(self.ds, branches)]
        x = torch.cat(branches, dim=1)
        x = self.pool1(x)
        x = torch.cat([conv(x) for conv in self.ms2], dim=1)
        return self.pool2(x)
