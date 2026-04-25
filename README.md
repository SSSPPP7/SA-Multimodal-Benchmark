# MDF-SACN

PyTorch implementation of MDF-SACN for hierarchical SA recognition and 5-min prediction from EEG and eye movement.

## Files

```text
MDF_SACN/
  configs/default.yaml
  data/README.md
  datasets/sa_dataset.py
  models/
  losses/joint_loss.py
  train.py
  evaluate.py
  loso_cv.py
  utils/
  requirements.txt
```

## Input

The training file is configured by `data.processed_npz` in `configs/default.yaml`.

Required arrays:

```text
eeg        [N, 32, 500]
em         [N, 6, 500]
subject_id [N]
probe_id   [N]
y_rec      [N, 3]
y_pred     [N, 3]
mask_pred  [N, 3]
```

Optional arrays:

```text
mask_rec    [N, 3]
time_sec    [N]
quality_eeg [N]
quality_em  [N]
```

The three label columns are ordered as SA1, SA2, SA3. High SA is encoded as 1.

## Run

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml --test-subject S01 --run-dir runs/fold_S01
python evaluate.py --checkpoint runs/fold_S01/best.pt --config configs/default.yaml --test-subject S01
python loso_cv.py --config configs/default.yaml --out-dir runs/loso
```

The repository does not include data files or stored experiment results.
