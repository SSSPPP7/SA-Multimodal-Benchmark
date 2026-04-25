import torch
import torch.nn as nn

LEVELS = ("sa1", "sa2", "sa3")


class HPTC(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.phi = nn.ModuleDict({
            l: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            )
            for l in LEVELS
        })
        self.g12 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.g23 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.rec = nn.ModuleDict({l: nn.Linear(hidden_dim, 1) for l in LEVELS})
        self.trans = nn.ModuleDict({
            l: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for l in LEVELS
        })
        self.tgate = nn.ModuleDict({l: nn.Linear(hidden_dim + 1, hidden_dim) for l in LEVELS})
        self.pred = nn.ModuleDict({l: nn.Linear(hidden_dim, 1) for l in LEVELS})

    def forward(self, fused):
        z1 = self.phi["sa1"](fused["sa1"])
        z2 = self.phi["sa2"](fused["sa2"])
        z3 = self.phi["sa3"](fused["sa3"])

        h1 = z1
        g12 = torch.sigmoid(self.g12(torch.cat([h1, z2], dim=1)))
        h2 = z2 + g12 * h1
        g23 = torch.sigmoid(self.g23(torch.cat([h2, z3], dim=1)))
        h3 = z3 + g23 * h2

        hs = {"sa1": h1, "sa2": h2, "sa3": h3}
        rec_logits = {l: self.rec[l](hs[l]).squeeze(-1) for l in LEVELS}
        pred_logits, temporal = {}, {}

        for l in LEVELS:
            delta = self.trans[l](hs[l])
            tau = torch.sigmoid(self.tgate[l](torch.cat([hs[l], rec_logits[l].unsqueeze(1)], dim=1)))
            temporal[l] = tau
            pred_logits[l] = self.pred[l](hs[l] + tau * delta).squeeze(-1)

        return {
            "rec_logits": rec_logits,
            "pred_logits": pred_logits,
            "gates": {"g12": g12, "g23": g23, "temporal": temporal},
            "features": {"h1": h1, "h2": h2, "h3": h3},
        }
