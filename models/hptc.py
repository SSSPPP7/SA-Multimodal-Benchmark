from __future__ import annotations

import torch
from torch import nn


class HPTC(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_levels: int = 3) -> None:
        super().__init__()
        if num_levels != 3:
            raise ValueError("MDF-SACN is defined for SA1, SA2 and SA3.")
        self.phi = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(3)])
        self.gate_12 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_23 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.rec_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])
        self.delta = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])
        self.time_gates = nn.ModuleList([nn.Linear(hidden_dim + 1, hidden_dim) for _ in range(3)])
        self.pred_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        u1 = self.phi[0](fused[:, 0])
        u2 = self.phi[1](fused[:, 1])
        u3 = self.phi[2](fused[:, 2])
        h1 = u1
        g12 = torch.sigmoid(self.gate_12(torch.cat([u1, u2], dim=-1)))
        h2 = u2 + g12 * u1
        g23 = torch.sigmoid(self.gate_23(torch.cat([h2, u3], dim=-1)))
        h3 = u3 + g23 * h2
        states = [h1, h2, h3]
        rec_logits = torch.cat([head(h) for head, h in zip(self.rec_heads, states)], dim=-1)
        pred_states, gt_values = [], []
        for i, h in enumerate(states):
            d = self.delta[i](h)
            gt = torch.sigmoid(self.time_gates[i](torch.cat([h, rec_logits[:, i:i+1]], dim=-1)))
            pred_states.append(h + gt * d)
            gt_values.append(gt)
        pred_logits = torch.cat([head(h) for head, h in zip(self.pred_heads, pred_states)], dim=-1)
        return {
            "rec_logits": rec_logits,
            "pred_logits": pred_logits,
            "states": torch.stack(states, dim=1),
            "pred_states": torch.stack(pred_states, dim=1),
            "gates": {"g12": g12, "g23": g23, "gt": torch.stack(gt_values, dim=1)},
        }
