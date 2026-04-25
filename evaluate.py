import argparse

import torch
from torch.utils.data import DataLoader

from datasets.sa_dataset import SADataset, loso_indices
from models.mdf_sacn import MDFSACN
from train import predict
from utils.io import load_yaml, save_json
from utils.metrics import collect_six_task_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test-subject")
    ap.add_argument("--out", default="eval_metrics.json")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    data_path = a.data or cfg["data"]["processed_npz"]
    ds = SADataset(data_path)
    if a.test_subject:
        _, _, te = loso_indices(
            ds.subject_id,
            a.test_subject,
            cfg["data"].get("val_ratio_subjects", 0.2),
            int(cfg.get("seed", 42)),
        )
        ds = SADataset(data_path, te)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDFSACN(cfg["model"]).to(device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    preds, labels, _, _ = predict(model, loader, device)
    metrics = collect_six_task_metrics(preds, labels, float(cfg["train"].get("threshold", 0.5)))
    save_json(metrics, a.out)
    print(metrics)


if __name__ == "__main__":
    main()
