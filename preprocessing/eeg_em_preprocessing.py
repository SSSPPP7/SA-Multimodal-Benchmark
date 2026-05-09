from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, resample_poly, sosfiltfilt


def bandpass_eeg(eeg: np.ndarray, fs: float = 256.0, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1)


def detect_eeg_artifacts(eeg: np.ndarray, fs: float = 256.0, jump_uv_per_ms: float = 90.0, flat_std_uv: float = 0.8) -> np.ndarray:

    eeg = np.asarray(eeg, dtype=float)
    mask = np.zeros_like(eeg, dtype=bool)
    diff = np.abs(np.diff(eeg, axis=-1)) * fs / 1000.0
    mask[..., 1:] |= diff > jump_uv_per_ms
    win = max(int(round(0.2 * fs)), 1)
    for ch in range(eeg.shape[0]):
        for start in range(0, eeg.shape[1] - win + 1):
            if np.std(eeg[ch, start:start + win]) < flat_std_uv:
                mask[ch, start:start + win] = True
    return mask


def resample_signal(x: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    from math import gcd
    g = gcd(orig_fs, target_fs)
    return resample_poly(x, target_fs // g, orig_fs // g, axis=-1)


def baseline_correct(eeg_window: np.ndarray, baseline_samples: int) -> np.ndarray:
    baseline = eeg_window[..., :baseline_samples].mean(axis=-1, keepdims=True)
    return eeg_window - baseline


def interpolate_short_gaps(x: np.ndarray, missing_mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    t = np.arange(out.shape[-1])
    for ch in range(out.shape[0]):
        mask = missing_mask[ch].astype(bool)
        if mask.all():
            continue
        start = 0
        while start < len(mask):
            if not mask[start]:
                start += 1
                continue
            end = start
            while end < len(mask) and mask[end]:
                end += 1
            if end - start <= max_gap:
                good = ~mask
                cs = CubicSpline(t[good], out[ch, good], bc_type="natural")
                out[ch, start:end] = cs(t[start:end])
            start = end
    return out
