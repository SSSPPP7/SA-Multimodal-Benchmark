from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

LEVELS = ("sa1", "sa2", "sa3")


class SADataset(Dataset):

    def __init__(self, npz_path: Union[str, Path], indices: Optional[Sequence[int]] = None, dtype=torch.float32):
        self.path = Path(npz_path)
        with np.load(self.path, allow_pickle=True) as z:
            self.eeg = np.asarray(z["eeg"], dtype=np.float32)
            self.em = np.asarray(z["em"], dtype=np.float32)
            n = len(self.eeg)
            self.subject_id = np.asarray(z["subject_id"] if "subject_id" in z else np.arange(n)).astype(str)
            self.probe_id = np.asarray(z["probe_id"] if "probe_id" in z else np.arange(n)).astype(str)
            if "time_sec" in z:
                self.time_sec = np.asarray(z["time_sec"], dtype=np.float32)
            elif "timestamp_sec" in z:
                self.time_sec = np.asarray(z["timestamp_sec"], dtype=np.float32)
            else:
                self.time_sec = np.zeros(n, dtype=np.float32)
            self.y_rec = self._load_level_array(z, "y_rec", default=0.0)
            self.y_pred = self._load_level_array(z, "y_pred", default=0.0)
            self.mask_pred = self._load_level_array(z, "mask_pred", default=0.0, alt_prefix="mask")
            self.mask_rec = self._load_level_array(z, "mask_rec", default=1.0)
            self.quality_eeg = np.asarray(z["quality_eeg"] if "quality_eeg" in z else np.ones(n), dtype=np.float32)
            self.quality_em = np.asarray(z["quality_em"] if "quality_em" in z else np.ones(n), dtype=np.float32)

        self._validate()
        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.eeg)), dtype=int)
        if self.indices.ndim != 1:
            raise ValueError("indices must be a 1D sequence")
        if len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= len(self.eeg)):
            raise IndexError("indices contain values outside dataset range")
        self.dtype = dtype

    @staticmethod
    def _load_level_array(z, key: str, default: float, alt_prefix: Optional[str] = None) -> np.ndarray:
        if key in z:
            arr = np.asarray(z[key], dtype=np.float32)
            if arr.ndim == 1:
                if arr.size % 3 != 0:
                    raise ValueError(f"{key} is 1D but length is not divisible by 3: {arr.shape}")
                arr = arr.reshape(-1, 3)
            return arr.astype(np.float32)
        prefix = alt_prefix if alt_prefix is not None else key
        cols = []
        n = len(z["eeg"])
        for lvl in LEVELS:
            col = f"{prefix}_{lvl}"
            cols.append(np.asarray(z[col], dtype=np.float32) if col in z else np.full(n, default, dtype=np.float32))
        return np.stack(cols, axis=1).astype(np.float32)

    def _validate(self) -> None:
        n = len(self.eeg)
        if self.eeg.ndim != 3 or self.eeg.shape[1:] != (32, 500):
            raise ValueError(f"EEG must have shape [N, 32, 500], got {self.eeg.shape}")
        if self.em.ndim != 3 or self.em.shape[1:] != (6, 500):
            raise ValueError(f"EM must have shape [N, 6, 500], got {self.em.shape}")
        for name, arr in {
            "y_rec": self.y_rec,
            "y_pred": self.y_pred,
            "mask_pred": self.mask_pred,
            "mask_rec": self.mask_rec,
        }.items():
            if arr.shape != (n, 3):
                raise ValueError(f"{name} must have shape [{n}, 3], got {arr.shape}")
        for name, arr in {"subject_id": self.subject_id, "probe_id": self.probe_id, "time_sec": self.time_sec,
                          "quality_eeg": self.quality_eeg, "quality_em": self.quality_em}.items():
            if len(arr) != n:
                raise ValueError(f"{name} length must be {n}, got {len(arr)}")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict:
        i = int(self.indices[idx])
        item = {
            "eeg": torch.as_tensor(self.eeg[i], dtype=self.dtype),
            "em": torch.as_tensor(self.em[i], dtype=self.dtype),
            "y_rec": torch.as_tensor(self.y_rec[i], dtype=self.dtype),
            "y_pred": torch.as_tensor(self.y_pred[i], dtype=self.dtype),
            "mask_pred": torch.as_tensor(self.mask_pred[i], dtype=self.dtype),
            "mask_rec": torch.as_tensor(self.mask_rec[i], dtype=self.dtype),
            "subject_id": self.subject_id[i],
            "probe_id": self.probe_id[i],
            "time_sec": torch.tensor(float(self.time_sec[i]), dtype=self.dtype),
            "quality_eeg": torch.tensor(float(self.quality_eeg[i]), dtype=self.dtype),
            "quality_em": torch.tensor(float(self.quality_em[i]), dtype=self.dtype),
        }
        for j, lvl in enumerate(LEVELS):
            item[f"y_rec_{lvl}"] = item["y_rec"][j]
            item[f"y_pred_{lvl}"] = item["y_pred"][j]
            item[f"mask_{lvl}"] = item["mask_pred"][j]
            item[f"mask_rec_{lvl}"] = item["mask_rec"][j]
        return item

    def get_subjects(self) -> np.ndarray:
        return np.unique(self.subject_id.astype(str))


def loso_indices(subject_id: Sequence, test_subject: str, val_ratio_subjects: float = 0.2,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = np.asarray(subject_id).astype(str)
    subjects = np.unique(sid)
    test_subject = str(test_subject)
    if test_subject not in subjects:
        raise ValueError(f"{test_subject} not in dataset subjects: {subjects.tolist()}")
    train_subjects = np.array([s for s in subjects if s != test_subject])
    rng = np.random.default_rng(seed)
    rng.shuffle(train_subjects)
    n_val = max(1, int(round(len(train_subjects) * val_ratio_subjects))) if len(train_subjects) > 1 else 0
    val_subjects = set(train_subjects[:n_val])
    train_subjects = set(train_subjects[n_val:])
    train_idx = np.flatnonzero(np.isin(sid, list(train_subjects)))
    val_idx = np.flatnonzero(np.isin(sid, list(val_subjects)))
    test_idx = np.flatnonzero(sid == test_subject)
    return train_idx, val_idx, test_idx
