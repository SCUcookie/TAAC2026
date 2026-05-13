# Eval 3 Best CPU Load Diagnostics

## Date

2026-05-12

## Purpose

Third evaluation task for 2026-05-12. Preserve the current best scoring path and add bounded diagnostics to understand evaluation data and score distribution.

## Eval Package

`submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`

## Base Result

Current best:

- AUC: `0.819544`
- inference time: `481.66s`
- evaluator: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

No scoring-path change:

- same model
- same checkpoint
- same CPU-staged checkpoint load
- same single-pass logits
- same predictions logic

Added bounded logs:

- `DIAG_CONFIG`
- `DIAG_DATASET`
- `DIAG_FIRST_BATCH`
- `DIAG_SEQ`
- `DIAG_TENSOR`
- `DIAG_PROGRESS`
- `DIAG_SCORE`
- `DIAG_SCORE_HIST`

## Expected Outcome

Expected AUC should match the current best within normal platform determinism. This run is mainly for confirmation and diagnostics after tail blend underperformed.

## Result

- AUC: `0.819544`
- predictions: `310000`
- unique users: `310000`
- internal diagnostic elapsed time: `434.885s`
- infer-stage wall time from log timestamps: about `469.77s`
- outcome: tied current best and produced useful diagnostics
