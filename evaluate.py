from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets import SADataset, loso_subject_split, make_loaders
from models import build_model_from_config
from train import choose_device, collect_predictions, evaluate_predictions
from utils import load_config
from utils.checkpoint import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--test-subject", required=True)
    p.add_argument("--output-dir", default="outputs/eval")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint, "cpu")
    config = load_config(args.config) if args.config else ckpt["config"]
    device = choose_device(config.get("train", {}).get("device", "auto"))
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    try:
        test_subject = int(args.test_subject)
    except ValueError:
        test_subject = args.test_subject

    dataset = SADataset(config["data"]["processed_npz"], config)
    val_cfg = config.get("validation", {})
    split = loso_subject_split(dataset.subject_id, test_subject, config.get("seed", 42), val_cfg.get("strategy", "subject_holdout"), int(val_cfg.get("num_val_subjects", 1)), val_cfg.get("explicit_val_subjects", []))
    _, _, test_loader = make_loaders(dataset, split, int(config.get("train", {}).get("batch_size", 32)), int(config.get("train", {}).get("num_workers", 0)))
    pred = collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(pred)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
