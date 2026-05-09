from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class JointMDFSACNLoss(nn.Module):

    def __init__(self, lambda_pred: float = 1.0, lambda_uni: float = 0.3, lambda_cg: float = 0.1, eps: float = 1e-8, detach_contribution: bool = False) -> None:
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_uni = lambda_uni
        self.lambda_cg = lambda_cg
        self.eps = eps
        self.detach_contribution = detach_contribution

    @staticmethod
    def _bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
        if mask is None:
            return loss.mean()
        mask = mask.float()
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _sum_levels(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        out = logits.new_zeros(())
        for l in range(logits.size(1)):
            out = out + self._bce(logits[:, l], target[:, l], None if mask is None else mask[:, l])
        return out

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_rec = batch["y_rec"].float()
        y_pred = batch["y_pred"].float()
        mask_rec = batch.get("mask_rec")
        mask_pred = batch.get("mask_pred")

        l_cur = self._sum_levels(outputs["rec_logits"], y_rec, mask_rec)
        l_pred = self._sum_levels(outputs["pred_logits"], y_pred, mask_pred)
        l_uni_eeg = self._sum_levels(outputs["eeg_aux_logits"], y_rec, mask_rec)
        l_uni_em = self._sum_levels(outputs["em_aux_logits"], y_rec, mask_rec)
        l_uni = l_uni_eeg + l_uni_em

        eeg_abs = outputs["eeg_aux_logits"].abs()
        em_abs = outputs["em_aux_logits"].abs()
        denom = eeg_abs + em_abs + self.eps
        contrib = torch.stack([eeg_abs / denom, em_abs / denom], dim=-1)
        if self.detach_contribution:
            contrib = contrib.detach()
        align = (outputs["fusion_weights"] - contrib).abs().sum(dim=-1)
        if mask_rec is not None:
            l_cg = (align * mask_rec.float()).sum() / mask_rec.float().sum().clamp_min(1.0)
        else:
            l_cg = align.mean()

        total = l_cur + self.lambda_pred * l_pred + self.lambda_uni * l_uni + self.lambda_cg * l_cg
        parts = {"total": total.detach(), "l_cur": l_cur.detach(), "l_pred": l_pred.detach(), "l_uni": l_uni.detach(), "l_uni_eeg": l_uni_eeg.detach(), "l_uni_em": l_uni_em.detach(), "l_cg": l_cg.detach()}
        return total, parts


def build_loss_from_config(config: dict) -> JointMDFSACNLoss:
    cfg = config.get("loss", {})
    return JointMDFSACNLoss(
        lambda_pred=cfg.get("lambda_pred", 1.0),
        lambda_uni=cfg.get("lambda_uni", 0.3),
        lambda_cg=cfg.get("lambda_cg", 0.1),
        eps=float(cfg.get("eps", 1e-8)),
        detach_contribution=bool(cfg.get("detach_contribution", False)),
    )
