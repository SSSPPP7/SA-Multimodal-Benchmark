from __future__ import annotations

import torch
from torch import nn


class HierarchicalDynamicFusion(nn.Module):

    def __init__(self, feature_dim: int = 32, level_emb_dim: int = 16, mlp_hidden_dim: int = 16, num_levels: int = 3) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.fused_dim = 2 * feature_dim
        self.num_levels = num_levels
        self.level_embedding = nn.Embedding(num_levels, level_emb_dim)
        self.weight_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * feature_dim + level_emb_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 2),
            )
            for _ in range(num_levels)
        ])

    def forward(self, f_eeg: torch.Tensor, f_em: torch.Tensor) -> dict[str, torch.Tensor]:
        h_eeg = f_eeg.mean(dim=-1)
        h_em = f_em.mean(dim=-1)
        batch = h_eeg.size(0)
        fused, weights = [], []
        for level, mlp in enumerate(self.weight_mlps):
            level_ids = torch.full((batch,), level, dtype=torch.long, device=h_eeg.device)
            emb = self.level_embedding(level_ids)
            w = torch.softmax(mlp(torch.cat([h_eeg, h_em, emb], dim=-1)), dim=-1)
            z = torch.cat([w[:, 0:1] * h_eeg, w[:, 1:2] * h_em], dim=-1)
            fused.append(z)
            weights.append(w)
        return {
            "fused": torch.stack(fused, dim=1),
            "weights": torch.stack(weights, dim=1),
            "h_eeg": h_eeg,
            "h_em": h_em,
        }
