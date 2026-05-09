from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from datasets import SADataset, loso_subject_split, make_loaders
from losses import build_loss_from_config
from models import build_model_from_config
from utils import load_config, save_config, set_seed, metrics_by_level
from utils.checkpoint import save_checkpoint, load_checkpoint


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["eeg"], batch["em"])
            loss, parts = criterion(outputs, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def collect_predictions(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    store: dict[str, list[np.ndarray]] = {k: [] for k in ["rec_logits", "pred_logits", "y_rec", "y_pred", "mask_pred", "mask_rec", "fusion_weights", "g12", "g23", "gt", "sample_index"]}
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            out = model(batch_dev["eeg"], batch_dev["em"])
            store["rec_logits"].append(out["rec_logits"].cpu().numpy())
            store["pred_logits"].append(out["pred_logits"].cpu().numpy())
            store["y_rec"].append(batch["y_rec"].numpy())
            store["y_pred"].append(batch["y_pred"].numpy())
            store["mask_pred"].append(batch["mask_pred"].numpy())
            store["mask_rec"].append(batch["mask_rec"].numpy())
            store["fusion_weights"].append(out["fusion_weights"].cpu().numpy())
            store["g12"].append(out["gates"]["g12"].cpu().numpy())
            store["g23"].append(out["gates"]["g23"].cpu().numpy())
            store["gt"].append(out["gates"]["gt"].cpu().numpy())
            store["sample_index"].append(batch["sample_index"].numpy())
    return {k: np.concatenate(v, axis=0) for k, v in store.items() if v}


def evaluate_predictions(pred: dict[str, np.ndarray]) -> dict[str, float]:
    metrics = {}
    metrics.update(metrics_by_level(pred["rec_logits"], pred["y_rec"], pred.get("mask_rec"), prefix="rec"))
    metrics.update(metrics_by_level(pred["pred_logits"], pred["y_pred"], pred.get("mask_pred"), prefix="pred"))
    return metrics


def train_fold(config: dict, test_subject: int | str, output_dir: str | Path) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    dataset = SADataset(config["data"]["processed_npz"], config)
    val_cfg = config.get("validation", {})
    split = loso_subject_split(
        dataset.subject_id,
        test_subject=test_subject,
        seed=int(config.get("seed", 42)),
        strategy=val_cfg.get("strategy", "subject_holdout"),
        num_val_subjects=int(val_cfg.get("num_val_subjects", 1)),
        explicit_val_subjects=val_cfg.get("explicit_val_subjects", []),
    )
    train_cfg = config.get("train", {})
    train_loader, val_loader, test_loader = make_loaders(dataset, split, int(train_cfg.get("batch_size", 32)), int(train_cfg.get("num_workers", 0)))

    device = choose_device(train_cfg.get("device", "auto"))
    model = build_model_from_config(config).to(device)
    criterion = build_loss_from_config(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
        eps=float(train_cfg.get("adam_eps", 1e-8)),
    )

    best_val = float("inf")
    patience = int(train_cfg.get("early_stop_patience", 20))
    bad = 0
    for epoch in range(1, int(train_cfg.get("epochs", 200)) + 1):
        train_parts = run_epoch(model, train_loader, criterion, device, optimizer)
        val_parts = run_epoch(model, val_loader, criterion, device, None)
        val_loss = val_parts.get("total", float("inf"))
        history_line = {"epoch": epoch, "train": train_parts, "val": val_parts}
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(history_line, ensure_ascii=False) + "\n")
        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, config, best_val)
        else:
            bad += 1
            if bad >= patience:
                break

    ckpt = load_checkpoint(out_dir / "best.pt", device)
    model.load_state_dict(ckpt["model_state"])
    pred = collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(pred)
    metrics["test_subject"] = int(test_subject) if isinstance(test_subject, (np.integer, int)) else str(test_subject)
    metrics["best_val_loss"] = float(best_val)
    metrics["n_train"] = int(len(split.train))
    metrics["n_val"] = int(len(split.val))
    metrics["n_test"] = int(len(split.test))
    if train_cfg.get("save_predictions", True):
        np.savez_compressed(out_dir / "test_predictions.npz", **pred)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--test-subject", required=True)
    p.add_argument("--output-dir", default="outputs/fold")
    args = p.parse_args()
    config = load_config(args.config)
    subject: int | str
    try:
        subject = int(args.test_subject)
    except ValueError:
        subject = args.test_subject
    metrics = train_fold(config, subject, args.output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
