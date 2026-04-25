import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ("sa1", "sa2", "sa3")


class HierarchicalDynamicFusion(nn.Module):
    def __init__(self, hidden_dim=64, level_emb_dim=16, mlp_hidden=16):
        super().__init__()
        self.level_emb = nn.Embedding(3, level_emb_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.mlps = nn.ModuleDict({
            l: nn.Sequential(
                nn.Linear(hidden_dim * 2 + level_emb_dim, mlp_hidden),
                nn.ELU(inplace=True),
                nn.Linear(mlp_hidden, 2),
            )
            for l in LEVELS
        })

    def forward(self, eeg_map, em_map):
        eeg = self.gap(eeg_map).squeeze(-1)
        em = self.gap(em_map).squeeze(-1)
        b = eeg.size(0)
        fused, weights = {}, {}
        for i, l in enumerate(LEVELS):
            emb = self.level_emb.weight[i].unsqueeze(0).expand(b, -1)
            alpha = F.softmax(self.mlps[l](torch.cat([eeg, em, emb], dim=1)), dim=1)
            fused[l] = alpha[:, 0:1] * eeg + alpha[:, 1:2] * em
            weights[l] = alpha
        return fused, weights
