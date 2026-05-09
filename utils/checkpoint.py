from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, config: dict, best_val: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
        "best_val": best_val,
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device)
