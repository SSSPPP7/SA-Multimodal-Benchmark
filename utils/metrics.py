from __future__ import annotations

import numpy as np
import torch


def binary_metrics_from_logits(logits: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor | None = None) -> dict[str, float]:
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    pred = (logits >= 0).astype(np.int64)
    target = target.astype(np.int64)
    if mask is not None:
        keep = mask.astype(bool)
        pred = pred[keep]
        target = target[keep]
    if target.size == 0:
        return {"acc": float("nan"), "recall": float("nan"), "precision": float("nan"), "f1": float("nan"), "n": 0}
    tp = int(((pred == 1) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-12)
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(target.size)}


def metrics_by_level(logits: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, prefix: str = "rec") -> dict[str, float]:
    out: dict[str, float] = {}
    for i, name in enumerate(["SA1", "SA2", "SA3"]):
        m = None if mask is None else mask[:, i]
        vals = binary_metrics_from_logits(logits[:, i], target[:, i], m)
        for k, v in vals.items():
            out[f"{prefix}_{name}_{k}"] = v
    return out
