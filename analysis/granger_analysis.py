from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

LEVELS = ["SA1", "SA2", "SA3"]


def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    n = len(series)
    return np.column_stack([series[lag - k:n - k] for k in range(1, lag + 1)])


def _rss(y: np.ndarray, x: np.ndarray) -> float:
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.sum(resid ** 2))


def granger_xy(x: np.ndarray, y: np.ndarray, lag: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) <= 2 * lag + 1:
        raise ValueError("not enough points for requested lag")
    y_target = y[lag:]
    y_lags = _lag_matrix(y, lag)
    x_lags = _lag_matrix(x, lag)
    rss_r = _rss(y_target, y_lags)
    rss_u = _rss(y_target, np.column_stack([y_lags, x_lags]))
    df1 = lag
    df2 = len(y_target) - 2 * lag - 1
    f_value = ((rss_r - rss_u) / df1) / max(rss_u / max(df2, 1), 1e-12)
    p_value = float(1.0 - f_dist.cdf(max(f_value, 0.0), df1, df2)) if df2 > 0 else float("nan")
    strength = (rss_r - rss_u) / max(rss_r, 1e-12)
    return {"F": float(f_value), "p": p_value, "G": float(strength), "rss_r": rss_r, "rss_u": rss_u, "lag": lag, "n": len(y_target)}


def adjust_pvalues(pvals: list[float], method: str = "fdr_bh") -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if method == "none":
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0).tolist()
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * p[idx]
            running = max(running, val)
            adjusted[idx] = min(running, 1.0)
        return adjusted.tolist()
    if method == "fdr_bh":
        running = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            true_rank = m - rank + 1
            val = p[idx] * m / true_rank
            running = min(running, val)
            adjusted[idx] = min(running, 1.0)
        return adjusted.tolist()
    raise ValueError(f"unknown correction method: {method}")


def granger_pairwise(scores: np.ndarray, max_lag: int = 5, correction: str = "fdr_bh") -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError(f"Expected scores [T,3], got {scores.shape}")
    rows = []
    for cause in range(3):
        for effect in range(3):
            if cause == effect:
                continue
            best = None
            for lag in range(1, max_lag + 1):
                vals = granger_xy(scores[:, cause], scores[:, effect], lag)
                vals.update({"cause": LEVELS[cause], "effect": LEVELS[effect]})
                if best is None or vals["p"] < best["p"]:
                    best = vals
            rows.append(best)
    q = adjust_pvalues([r["p"] for r in rows], correction)
    for r, qv in zip(rows, q):
        r["p_corrected"] = qv
    return pd.DataFrame(rows)[["cause", "effect", "lag", "F", "p", "p_corrected", "G", "rss_r", "rss_u", "n"]]


def local_granger(scores: np.ndarray, window: int, step: int, max_lag: int = 5, correction: str = "fdr_bh") -> pd.DataFrame:
    frames = []
    for start in range(0, len(scores) - window + 1, step):
        df = granger_pairwise(scores[start:start + window], max_lag=max_lag, correction=correction)
        df.insert(0, "start", start)
        df.insert(1, "end", start + window)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_scores(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        arr = data["rec_scores"] if "rec_scores" in data else data["scores"]
        return arr[:, :3]
    df = pd.read_csv(path)
    return df[["SA1", "SA2", "SA3"]].to_numpy(float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="CSV with SA1,SA2,SA3 columns or NPZ from sliding_window_inference")
    p.add_argument("--output", default="outputs/granger_results.csv")
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--correction", default="fdr_bh", choices=["none", "bonferroni", "holm", "fdr_bh"])
    p.add_argument("--local-window", type=int, default=0)
    p.add_argument("--local-step", type=int, default=1)
    args = p.parse_args()
    scores = _load_scores(args.scores)
    if args.local_window > 0:
        df = local_granger(scores, args.local_window, args.local_step, args.max_lag, args.correction)
    else:
        df = granger_pairwise(scores, args.max_lag, args.correction)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
