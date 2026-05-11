# 2026-05-11 1545 Eval Tail Blend New Baseline

## Purpose

Evaluation-only attempt to improve the new 2-GPU model baseline:

- Baseline AUC: `0.818178`
- Baseline inference time: about `400s`
- Baseline evaluator: archived/original single-pass evaluator

## Candidate Evaluator

Path: `submission/runs/2026-05-11_1545_eval-tail-blend-new-baseline/eval/`

This package keeps the normal prefix/full-window forward pass and adds a true raw-sequence tail-window forward pass in the same Parquet read.

Default behavior:

- `EVAL_TAIL_BLEND=true`
- `EVAL_TAIL_BLEND_WEIGHT=0.15`
- final logit = `0.85 * prefix_logit + 0.15 * tail_logit`

The tail view uses `values[end - use_len:end]` for sequence side-info and timestamps, while the normal view remains `values[start:start + use_len]`.

## Risk

Expected runtime is roughly 2x the archived single-pass baseline because there are two model forwards per batch. This should still be far below the earlier TTA runs and within the platform budget.

## Checkpoint

Evaluate against the current best 2-GPU checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

Full platform path from logs:

`/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model`

## Package

`submission/platform_uploads/2026-05-11_eval_tail-blend-w015-new-baseline.zip`
