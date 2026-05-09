from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def build_sa_labels(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:

    required = {"subject_id", "sa_level", "task_template", "acc", "rt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    out["rt_z"] = np.nan
    out["sa_label"] = 0
    for level, level_df in out.groupby("sa_level"):
        correct_idx = level_df.index[level_df["acc"].astype(int) == 1]
        if len(correct_idx) == 0:
            continue
        z_parts = []
        for (_, _), g in out.loc[correct_idx].groupby(["subject_id", "task_template"]):
            mu = g["rt"].mean()
            sigma = g["rt"].std(ddof=0)
            z = (g["rt"] - mu) / (sigma + eps)
            z_parts.append(z)
        z_all = pd.concat(z_parts).sort_index()
        out.loc[z_all.index, "rt_z"] = z_all
        values = z_all.to_numpy().reshape(-1, 1)
        if len(values) < 2:
            out.loc[z_all.index, "sa_label"] = 1
            continue
        gmm = GaussianMixture(n_components=2, random_state=42)
        comp = gmm.fit_predict(values)
        fast_component = int(np.argmin(gmm.means_.reshape(-1)))
        out.loc[z_all.index, "sa_label"] = (comp == fast_component).astype(int)
    out.loc[out["acc"].astype(int) == 0, "sa_label"] = 0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    labels = build_sa_labels(pd.read_csv(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
