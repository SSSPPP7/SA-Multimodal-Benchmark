import torch.nn as nn

from .cross_modal_complement import CrossModalComplement
from .eeg_encoder import EEGEncoder
from .eye_encoder import EyeEncoder
from .hierarchical_dynamic_fusion import HierarchicalDynamicFusion
from .hptc import HPTC

LEVELS = ("sa1", "sa2", "sa3")


class MDFSACN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = int(cfg.get("hidden_dim", 64))
        drop = float(cfg.get("dropout", 0.2))
        self.eeg_encoder = EEGEncoder(
            int(cfg.get("eeg_channels", 32)),
            h,
            int(cfg.get("eeg_branch_channels", 16)),
            cfg.get("eeg_kernels_block1", [64, 32, 16, 8]),
            cfg.get("eeg_kernels_block3", [16, 8, 4, 2]),
            drop,
        )
        self.eye_encoder = EyeEncoder(
            int(cfg.get("em_channels", 6)),
            h,
            int(cfg.get("eye_conv_channels", 64)),
            drop,
        )
        self.ccm = CrossModalComplement(h, int(cfg.get("cross_attn_heads", 1)), drop)
        self.fusion = HierarchicalDynamicFusion(
            h,
            int(cfg.get("level_emb_dim", 16)),
            int(cfg.get("fusion_mlp_hidden", 16)),
        )
        self.hptc = HPTC(h, int(cfg.get("hptc_hidden_dim", h)), drop)
        self.eeg_aux = nn.ModuleDict({l: nn.Linear(h, 1) for l in LEVELS})
        self.em_aux = nn.ModuleDict({l: nn.Linear(h, 1) for l in LEVELS})
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, eeg, em):
        eeg_map, eeg_pool = self.eeg_encoder(eeg)
        em_map, em_pool = self.eye_encoder(em)
        eeg_aux = {l: self.eeg_aux[l](eeg_pool).squeeze(-1) for l in LEVELS}
        em_aux = {l: self.em_aux[l](em_pool).squeeze(-1) for l in LEVELS}
        eeg_enhanced, em_enhanced, attn = self.ccm(eeg_map, em_map)
        fused, fusion_weights = self.fusion(eeg_enhanced, em_enhanced)
        h = self.hptc(fused)
        return {
            "rec_logits": h["rec_logits"],
            "pred_logits": h["pred_logits"],
            "eeg_aux_logits": eeg_aux,
            "em_aux_logits": em_aux,
            "fusion_weights": fusion_weights,
            "gates": h["gates"],
            "features": h["features"],
            "attention_maps": attn,
        }
