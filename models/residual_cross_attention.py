from __future__ import annotations

import math
import torch
from torch import nn


class ResidualCrossAttentionModule(nn.Module):


    def __init__(self, feature_dim: int = 32, attn_dim: int = 64, num_heads: int = 1) -> None:
        super().__init__()
        if num_heads != 1:
            raise ValueError("The paper uses single-head cross-attention; keep cross_attn_heads=1.")
        self.q_eeg = nn.Linear(feature_dim, attn_dim, bias=False)
        self.k_eeg = nn.Linear(feature_dim, attn_dim, bias=False)
        self.v_eeg = nn.Linear(feature_dim, attn_dim, bias=False)
        self.q_em = nn.Linear(feature_dim, attn_dim, bias=False)
        self.k_em = nn.Linear(feature_dim, attn_dim, bias=False)
        self.v_em = nn.Linear(feature_dim, attn_dim, bias=False)
        self.out_eeg = nn.Linear(attn_dim, feature_dim, bias=False)
        self.out_em = nn.Linear(attn_dim, feature_dim, bias=False)

    @staticmethod
    def _attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, f_eeg: torch.Tensor, f_em: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if f_eeg.shape != f_em.shape:
            raise ValueError(f"RCAM expects matched shapes, got {tuple(f_eeg.shape)} and {tuple(f_em.shape)}")
        eeg = f_eeg.transpose(1, 2)  # [B,T,C]
        em = f_em.transpose(1, 2)
        eeg_comp = self._attn(self.q_eeg(eeg), self.k_em(em), self.v_em(em))
        em_comp = self._attn(self.q_em(em), self.k_eeg(eeg), self.v_eeg(eeg))
        eeg_out = self.out_eeg(eeg_comp) + eeg
        em_out = self.out_em(em_comp) + em
        return eeg_out.transpose(1, 2), em_out.transpose(1, 2)



CrossModalComplement = ResidualCrossAttentionModule
