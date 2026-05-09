from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray
    test_subject: int | str


class SADataset(Dataset):

    def __init__(self, npz_path: str | Path, config: dict | None = None) -> None:
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Processed data not found: {self.path}")
        cfg = (config or {}).get("data", {})
        data = np.load(self.path, allow_pickle=True)
        self.eeg = data[cfg.get("eeg_key", "eeg")].astype(np.float32)
        self.em = data[cfg.get("em_key", "em")].astype(np.float32)
        self.subject_id = data[cfg.get("subject_key", "subject_id")]
        probe_key = cfg.get("probe_key", "probe_id")
        self.probe_id = data[probe_key] if probe_key in data else np.arange(len(self.eeg))
        self.y_rec = data[cfg.get("y_rec_key", "y_rec")].astype(np.float32)
        self.y_pred = data[cfg.get("y_pred_key", "y_pred")].astype(np.float32)
        self.mask_pred = data[cfg.get("mask_pred_key", "mask_pred")].astype(np.float32)
        mask_key = cfg.get("mask_rec_key", "mask_rec")
        self.mask_rec = data[mask_key].astype(np.float32) if mask_key in data else np.ones_like(self.y_rec, dtype=np.float32)
        self._validate(cfg)

    def _validate(self, cfg: dict) -> None:
        n = self.eeg.shape[0]
        eeg_channels = int(cfg.get("eeg_channels", 32))
        em_channels = int(cfg.get("em_channels", 6))
        levels = int(cfg.get("num_levels", 3))
        if self.eeg.ndim != 3 or self.eeg.shape[1] != eeg_channels:
            raise ValueError(f"Expected eeg [N,{eeg_channels},T], got {self.eeg.shape}")
        if self.em.ndim != 3 or self.em.shape[1] != em_channels:
            raise ValueError(f"Expected em [N,{em_channels},T], got {self.em.shape}")
        for name, arr in [("subject_id", self.subject_id), ("y_rec", self.y_rec), ("y_pred", self.y_pred), ("mask_pred", self.mask_pred), ("mask_rec", self.mask_rec)]:
            if arr.shape[0] != n:
                raise ValueError(f"{name} first dimension {arr.shape[0]} != {n}")
        for name, arr in [("y_rec", self.y_rec), ("y_pred", self.y_pred), ("mask_pred", self.mask_pred), ("mask_rec", self.mask_rec)]:
            if arr.ndim != 2 or arr.shape[1] != levels:
                raise ValueError(f"Expected {name} [N,{levels}], got {arr.shape}")

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        subject = self.subject_id[idx]
        subject_tensor = torch.tensor(-1, dtype=torch.long) if isinstance(subject, str) else torch.tensor(subject, dtype=torch.long)
        return {
            "eeg": torch.from_numpy(self.eeg[idx]),
            "em": torch.from_numpy(self.em[idx]),
            "y_rec": torch.from_numpy(self.y_rec[idx]),
            "y_pred": torch.from_numpy(self.y_pred[idx]),
            "mask_pred": torch.from_numpy(self.mask_pred[idx]),
            "mask_rec": torch.from_numpy(self.mask_rec[idx]),
            "subject_id": subject_tensor,
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.subject_id)


def loso_subject_split(subject_id: np.ndarray, test_subject: int | str, seed: int = 42, strategy: str = "subject_holdout", num_val_subjects: int = 1, explicit_val_subjects: Sequence[int | str] | None = None) -> SplitIndices:
    subjects = np.unique(subject_id)
    if test_subject not in set(subjects.tolist()):
        raise ValueError(f"test_subject={test_subject!r} not found in subjects")
    trainval = np.array([s for s in subjects if s != test_subject], dtype=subjects.dtype)
    if explicit_val_subjects:
        val_subjects = np.array(list(explicit_val_subjects), dtype=subjects.dtype)
    elif strategy == "subject_holdout" and num_val_subjects > 0:
        rng = np.random.default_rng(seed)
        shuffled = trainval.copy()
        rng.shuffle(shuffled)
        val_subjects = np.sort(shuffled[:num_val_subjects])
    else:
        val_subjects = np.array([], dtype=subjects.dtype)
    train_subjects = np.array([s for s in trainval if s not in set(val_subjects.tolist())], dtype=subjects.dtype)
    if len(train_subjects) == 0:
        raise ValueError("No training subjects remain after validation split")
    if len(val_subjects) == 0:
        val_subjects = train_subjects.copy()
    return SplitIndices(
        train=np.flatnonzero(np.isin(subject_id, train_subjects)),
        val=np.flatnonzero(np.isin(subject_id, val_subjects)),
        test=np.flatnonzero(subject_id == test_subject),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subject=test_subject,
    )


def make_loaders(dataset: SADataset, split: SplitIndices, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(Subset(dataset, split.train), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(Subset(dataset, split.val), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(Subset(dataset, split.test), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
