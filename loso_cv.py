from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets import SADataset
from train import train_fold
from utils import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-dir", default="outputs/loso")
    args = p.parse_args()

    config = load_config(args.config)
    dataset = SADataset(config["data"]["processed_npz"], config)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in dataset.subjects:
        name = str(subject)
        print(f"[LOSO] test subject {name}")
        metrics = train_fold(config, subject.item() if hasattr(subject, "item") else subject, out_root / f"subject_{name}")
        rows.append(metrics)
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not k.startswith("n_") and k != "test_subject"] if rows else []
    summary = {f"mean_{k}": float(np.nanmean([r[k] for r in rows])) for k in numeric_keys}
    summary.update({f"std_{k}": float(np.nanstd([r[k] for r in rows])) for k in numeric_keys})
    with open(out_root / "loso_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"folds": rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
