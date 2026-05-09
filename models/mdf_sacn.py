from __future__ import annotations

import torch
from torch import nn

from .common import he_init
from .residual_cross_attention import ResidualCrossAttentionModule
from .eeg_encoder import EEGEncoder
from .eye_encoder import EyeEncoder
from .hierarchical_dynamic_fusion import HierarchicalDynamicFusion
from .hptc import HPTC


class MDFSACN(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 32,
        em_channels: int = 6,
        feature_dim: int = 32,
        dropout: float = 0.2,
        cross_attn_dim: int = 64,
        cross_attn_heads: int = 1,
        level_emb_dim: int = 16,
        fusion_mlp_hidden: int = 16,
        hptm_hidden_dim: int = 64,
        num_levels: int = 3,
    ) -> None:
        super().__init__()
        if feature_dim != 32:
            raise ValueError("Table 1 fixes feature_dim=32 for both EEG and EM branches.")
        self.eeg_encoder = EEGEncoder(eeg_channels, dropout)
        self.em_encoder = EyeEncoder(em_channels, dropout)
        self.rcam = ResidualCrossAttentionModule(feature_dim, cross_attn_dim, cross_attn_heads)
        self.hdfm = HierarchicalDynamicFusion(feature_dim, level_emb_dim, fusion_mlp_hidden, num_levels)
        self.hptm = HPTC(input_dim=2 * feature_dim, hidden_dim=hptm_hidden_dim, num_levels=num_levels)
        self.eeg_aux_head = nn.Linear(feature_dim, num_levels)
        self.em_aux_head = nn.Linear(feature_dim, num_levels)
        he_init(self)

    def forward(self, eeg: torch.Tensor, em: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        f_eeg = self.eeg_encoder(eeg)
        f_em = self.em_encoder(em)
        if f_eeg.size(-1) != f_em.size(-1):
            min_t = min(f_eeg.size(-1), f_em.size(-1))
            f_eeg = f_eeg[..., :min_t]
            f_em = f_em[..., :min_t]
        f_eeg, f_em = self.rcam(f_eeg, f_em)
        fusion = self.hdfm(f_eeg, f_em)
        hptm = self.hptm(fusion["fused"])
        return {
            "rec_logits": hptm["rec_logits"],
            "pred_logits": hptm["pred_logits"],
            "eeg_aux_logits": self.eeg_aux_head(fusion["h_eeg"]),
            "em_aux_logits": self.em_aux_head(fusion["h_em"]),
            "fusion_weights": fusion["weights"],
            "gates": hptm["gates"],
            "states": hptm["states"],
            "pred_states": hptm["pred_states"],
        }


def build_model_from_config(config: dict) -> MDFSACN:
    cfg = config.get("model", {})
    return MDFSACN(
        eeg_channels=cfg.get("eeg_channels", 32),
        em_channels=cfg.get("em_channels", 6),
        feature_dim=cfg.get("feature_dim", 32),
        dropout=cfg.get("dropout", 0.2),
        cross_attn_dim=cfg.get("cross_attn_dim", 64),
        cross_attn_heads=cfg.get("cross_attn_heads", 1),
        level_emb_dim=cfg.get("level_emb_dim", 16),
        fusion_mlp_hidden=cfg.get("fusion_mlp_hidden", 16),
        hptm_hidden_dim=cfg.get("hptm_hidden_dim", 64),
        num_levels=cfg.get("num_levels", 3),
    )
