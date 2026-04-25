from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.sa_dataset import SADataset, loso_indices
from losses.joint_loss import JointMDFSACNLoss
from models.mdf_sacn import MDFSACN
from utils.io import ensure_dir, load_yaml, save_json, save_yaml
from utils.logger import CSVLogger, get_logger
from utils.metrics import collect_six_task_metrics
from utils.seed import seed_worker, set_seed

LEVELS = ("sa1", "sa2", "sa3")


def device_from_cfg(cfg: dict) -> torch.device:
    device = cfg.get("train", {}).get("device", "auto")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def make_loader(ds: SADataset, batch_size: int, shuffle: bool, cfg: dict) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=bool(cfg.get("data", {}).get("pin_memory", True)),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(model: MDFSACN, loader: DataLoader, criterion: JointMDFSACNLoss,
              optimizer: Optional[torch.optim.Optimizer], device: torch.device,
              train: bool = True, clip: float = 0.0) -> dict:
    model.train(train)
    sums = {key: 0.0 for key in ["loss_total", "loss_rec", "loss_pred", "loss_aux", "loss_cg"]}
    n_samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch["eeg"], batch["em"])
            losses = criterion(outputs, batch)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        bs = int(batch["eeg"].size(0))
        n_samples += bs
        for key in sums:
            sums[key] += float(losses[key].detach().cpu()) * bs
    return {key: value / max(n_samples, 1) for key, value in sums.items()}


@torch.no_grad()
def predict(model: MDFSACN, loader: DataLoader, device: torch.device):
    model.eval()
    outputs = {"rec_logits": {lvl: [] for lvl in LEVELS}, "pred_logits": {lvl: [] for lvl in LEVELS}}
    labels = {key: [] for key in ["y_rec", "y_pred", "mask_pred", "mask_rec"]}
    meta = {"subject_id": [], "probe_id": [], "time_sec": [], "quality_eeg": [], "quality_em": []}
    extra = {
        "fusion_weights": {lvl: [] for lvl in LEVELS},
        "eeg_aux_logits": {lvl: [] for lvl in LEVELS},
        "em_aux_logits": {lvl: [] for lvl in LEVELS},
        "g12": [],
        "g23": [],
    }
    for batch in loader:
        batch_device = move_batch(batch, device)
        out = model(batch_device["eeg"], batch_device["em"])
        for lvl in LEVELS:
            outputs["rec_logits"][lvl].append(out["rec_logits"][lvl].detach().cpu().numpy())
            outputs["pred_logits"][lvl].append(out["pred_logits"][lvl].detach().cpu().numpy())
            extra["fusion_weights"][lvl].append(out["fusion_weights"][lvl].detach().cpu().numpy())
            extra["eeg_aux_logits"][lvl].append(out["eeg_aux_logits"][lvl].detach().cpu().numpy())
            extra["em_aux_logits"][lvl].append(out["em_aux_logits"][lvl].detach().cpu().numpy())
        extra["g12"].append(out["gates"]["g12"].detach().cpu().numpy())
        extra["g23"].append(out["gates"]["g23"].detach().cpu().numpy())
        for key in labels:
            labels[key].append(batch[key].numpy())
        meta["subject_id"].extend(batch["subject_id"])
        meta["probe_id"].extend(batch["probe_id"])
        meta["time_sec"].append(batch["time_sec"].numpy())
        meta["quality_eeg"].append(batch["quality_eeg"].numpy())
        meta["quality_em"].append(batch["quality_em"].numpy())

    for group in outputs:
        for lvl in LEVELS:
            outputs[group][lvl] = np.concatenate(outputs[group][lvl])
    for key in labels:
        labels[key] = np.concatenate(labels[key])
    for key in ["time_sec", "quality_eeg", "quality_em"]:
        meta[key] = np.concatenate(meta[key])
    for lvl in LEVELS:
        extra["fusion_weights"][lvl] = np.concatenate(extra["fusion_weights"][lvl])
        extra["eeg_aux_logits"][lvl] = np.concatenate(extra["eeg_aux_logits"][lvl])
        extra["em_aux_logits"][lvl] = np.concatenate(extra["em_aux_logits"][lvl])
    extra["g12"] = np.concatenate(extra["g12"])
    extra["g23"] = np.concatenate(extra["g23"])
    return outputs, labels, meta, extra


def train_one_fold(cfg: dict, test_subject: str, run_dir: Union[str, Path]) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    run_dir = ensure_dir(run_dir)
    logger = get_logger("MDF_SACN", str(run_dir / "train.log"))
    save_yaml(cfg, str(run_dir / "config.yaml"))

    full = SADataset(cfg["data"]["processed_npz"])
    train_idx, val_idx, test_idx = loso_indices(full.subject_id, test_subject,
                                                cfg["data"].get("val_ratio_subjects", 0.2),
                                                int(cfg.get("seed", 42)))
    train_ds = SADataset(cfg["data"]["processed_npz"], train_idx)
    val_ds = SADataset(cfg["data"]["processed_npz"], val_idx)
    test_ds = SADataset(cfg["data"]["processed_npz"], test_idx)

    device = device_from_cfg(cfg)
    model = MDFSACN(cfg["model"]).to(device)
    criterion = JointMDFSACNLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])),
        eps=float(cfg["train"].get("adam_eps", 1e-8)),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = make_loader(train_ds, batch_size, True, cfg)
    val_loader = make_loader(val_ds, batch_size, False, cfg)
    test_loader = make_loader(test_ds, batch_size, False, cfg)

    csv_logger = CSVLogger(str(run_dir / "epoch_log.csv"),
                           ["epoch", "train_loss", "val_loss", "loss_rec", "loss_pred", "loss_aux", "loss_cg"])
    best = float("inf")
    best_epoch = -1
    wait = 0
    patience = int(cfg["train"]["early_stop_patience"])
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_losses = run_epoch(model, train_loader, criterion, optimizer, device, True,
                                 float(cfg["train"].get("grad_clip_norm", 0.0)))
        val_losses = run_epoch(model, val_loader, criterion, None, device, False)
        csv_logger.log({"epoch": epoch, "train_loss": train_losses["loss_total"],
                        "val_loss": val_losses["loss_total"], **val_losses})
        logger.info(f"fold={test_subject} epoch={epoch} train={train_losses['loss_total']:.4f} "
                    f"val={val_losses['loss_total']:.4f}")
        if val_losses["loss_total"] < best:
            best = val_losses["loss_total"]
            best_epoch = epoch
            wait = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "test_subject": test_subject,
                        "best_val": best}, run_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    preds, labels, meta, extra = predict(model, test_loader, device)
    metrics = collect_six_task_metrics(preds, labels, float(cfg["train"].get("threshold", 0.5)))
    result = {"test_subject": test_subject, "best_epoch": best_epoch, "best_val": best, "metrics": metrics}
    save_json(result, str(run_dir / "test_metrics.json"))

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        **{f"rec_{lvl}": preds["rec_logits"][lvl] for lvl in LEVELS},
        **{f"pred_{lvl}": preds["pred_logits"][lvl] for lvl in LEVELS},
        **{f"eeg_aux_{lvl}": extra["eeg_aux_logits"][lvl] for lvl in LEVELS},
        **{f"em_aux_{lvl}": extra["em_aux_logits"][lvl] for lvl in LEVELS},
        y_rec=labels["y_rec"], y_pred=labels["y_pred"], mask_pred=labels["mask_pred"], mask_rec=labels["mask_rec"],
        subject_id=np.asarray(meta["subject_id"]), probe_id=np.asarray(meta["probe_id"]), time_sec=meta["time_sec"],
        quality_eeg=meta["quality_eeg"], quality_em=meta["quality_em"],
        g12=extra["g12"], g23=extra["g23"],
        **{f"fusion_{lvl}": extra["fusion_weights"][lvl] for lvl in LEVELS},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train_one_fold(cfg, args.test_subject, args.run_dir or str(Path(cfg["output_dir"]) / f"fold_{args.test_subject}"))


if __name__ == "__main__":
    main()
