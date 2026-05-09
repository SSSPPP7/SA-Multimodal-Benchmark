from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import build_model_from_config
from utils import load_config
from utils.checkpoint import load_checkpoint

def _windows(arr: np.ndarray, window_size: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 2:
        raise ValueError(f"Expected continuous array [C,T], got {arr.shape}")
    starts = np.arange(0, arr.shape[1] - window_size + 1, step_size, dtype=int)
    if len(starts) == 0:
        raise ValueError("Continuous signal is shorter than the requested window size")
    return np.stack([arr[:, s:s + window_size] for s in starts], axis=0).astype(np.float32), starts


def sliding_window_scores(model, eeg_cont: np.ndarray, em_cont: np.ndarray, window_size: int = 500, step_size: int = 100, batch_size: int = 64, device: str | torch.device = "cpu") -> dict[str, np.ndarray]:
    eeg_w, starts = _windows(eeg_cont, window_size, step_size)
    em_w, starts_em = _windows(em_cont, window_size, step_size)
    if not np.array_equal(starts, starts_em):
        raise ValueError("EEG and EM windows are not aligned")
    device = torch.device(device)
    model.eval().to(device)
    rec, pred, weights, g12, g23 = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            eeg = torch.from_numpy(eeg_w[i:i + batch_size]).to(device)
            em = torch.from_numpy(em_w[i:i + batch_size]).to(device)
            out = model(eeg, em)
            rec.append(out["rec_logits"].cpu().numpy())
            pred.append(out["pred_logits"].cpu().numpy())
            weights.append(out["fusion_weights"].cpu().numpy())
            g12.append(out["gates"]["g12"].mean(dim=-1).cpu().numpy())
            g23.append(out["gates"]["g23"].mean(dim=-1).cpu().numpy())
    return {
        "starts": starts,
        "centers": starts + window_size // 2,
        "rec_scores": np.concatenate(rec, axis=0),
        "pred_scores": np.concatenate(pred, axis=0),
        "fusion_weights": np.concatenate(weights, axis=0),
        "gate_12": np.concatenate(g12, axis=0),
        "gate_23": np.concatenate(g23, axis=0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--continuous-npz", required=True)
    p.add_argument("--eeg-key", default="eeg")
    p.add_argument("--em-key", default="em")
    p.add_argument("--output", default="outputs/sliding_scores.npz")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint, "cpu")
    config = load_config(args.config) if args.config else ckpt["config"]
    model = build_model_from_config(config)
    model.load_state_dict(ckpt["model_state"])
    data = np.load(args.continuous_npz)
    sw_cfg = config.get("analysis", {}).get("sliding_window", {})
    res = sliding_window_scores(
        model,
        data[args.eeg_key],
        data[args.em_key],
        window_size=int(sw_cfg.get("window_size", 500)),
        step_size=int(sw_cfg.get("step_size", 100)),
        batch_size=int(sw_cfg.get("batch_size", 64)),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    csv = out.with_suffix(".csv")
    mat = np.column_stack([res["centers"], res["rec_scores"]])
    np.savetxt(csv, mat, delimiter=",", header="center,SA1,SA2,SA3", comments="")
    print(f"saved {out} and {csv}")


if __name__ == "__main__":
    main()
