# Eval 3 Best CPU Load Diagnostics

## Date

2026-05-12

## Context

Evaluation 2 with tail blend weight `0.10` completed but did not improve:

- AUC `0.819304`
- inference time `641.82s`
- delta vs best: `-0.000240`

Current best remains:

- AUC `0.819544`
- inference time `481.66s`
- package `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`

## Upload

Submit:

`submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`

## Goal

Use the final evaluation slot to confirm the current best path and collect bounded diagnostic information from the official eval data. This should make fuller use of the 1000-line platform log without changing predictions.

## Expected Diagnostic Lines

- `DIAG_CONFIG`
- `DIAG_DATASET`
- `DIAG_FIRST_BATCH`
- `DIAG_SEQ`
- `DIAG_TENSOR`
- `DIAG_PROGRESS`
- `DIAG_SCORE`
- `DIAG_SCORE_HIST`

## Decision

Platform result:

- AUC `0.819544`
- predictions `310000`
- unique users `310000`
- internal diagnostic elapsed time `434.885s`
- infer-stage wall time from log timestamps about `469.77s`
- official inference time: not provided in the user message

The diagnostic package tied the current best AUC exactly while producing useful distribution logs. Keep it as the preferred observable package, with the non-diagnostic CPU-load package as the simpler fallback.

Key diagnostic observations:

- Score distribution: mean `0.1088370721`, std `0.1478978372`, p50 `0.0567224044`, p95 `0.3917785257`, p99 `0.8187775314`.
- Histogram counts by 0.1 bins: `[205120, 54735, 23703, 11647, 5833, 3168, 1558, 952, 1058, 2226]`.
- First batch sequence lengths are heavily capped: `seq_a` p50/p90/max `256/256/256`, `seq_b` `256/256/256`, `seq_c` p90/max `512/512`, `seq_d` p50/p90/max `512/512/512`.
- All first-batch non-empty tensors logged finite values; `item_dense_feats` has shape `(256, 0)`, so its finite rate is `nan` by empty tensor arithmetic and not a data issue.
