# SA-Multimodal-Benchmark

Paper-faithful implementation skeleton for **MDF-SACN: A Dynamic EEG-Eye Movement Fusion Framework for Progressive Modeling and Anticipatory Prediction of Hierarchical Situation Awareness**.

## Module names

The model contains the following sub-modules, named to match the latest
manuscript:

- **EEGEncoder / EyeEncoder** — dual-branch feature extractors (paper Table 1).
- **ResidualCrossAttentionModule (RCAM)** — bidirectional residual cross-attention
  for EEG–EM interaction (paper Eqs. 10–13). The pre-revision name was
  *Cross-modal Complementary Module (CCM)*; the original name is preserved as
  a backward-compatible alias `CrossModalComplement`.
- **HierarchicalDynamicFusion (HDFM)** — level-specific dynamic weighted fusion
  via weighted concatenation (paper Eqs. 14–16).
- **HPTC** — hierarchical progressive–temporal transfer module (paper Eqs.
  17–22), covering both current-state recognition and 5-min-ahead prediction.


## Expected processed data

Required arrays:

- `eeg`: `[N, 32, 500]`
- `em`: `[N, 6, 500]`
- `subject_id`: `[N]`
- `probe_id`: `[N]`
- `y_rec`: `[N, 3]`
- `y_pred`: `[N, 3]`
- `mask_pred`: `[N, 3]`
- optional `mask_rec`: `[N, 3]`

