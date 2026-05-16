# Eval Seed 20260512 Recent Crop TTA W004 CPU Load

## Date

2026-05-14

## Purpose

Retune yesterday's recent-crop TTA evaluator with a smaller blend weight while keeping the proven 20260512 checkpoint and CPU-staged load path fixed.

## Eval Package

`submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`

## Base Package

Copied from:

`submission/runs/2026-05-13_0006_eval-seed20260512-recent-crop-tta-cpuload/`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes

- keep CPU-staged checkpoint loading
- keep the same model weights and schema
- keep one extra recency-cropped forward pass
- reduce default logit blend weight from `0.08` to `0.04`
- keep crop lengths:
  - `seq_a:128`
  - `seq_b:128`
  - `seq_c:256`
  - `seq_d:256`
- no tail blend, target blend, multiview, or absolute-time-context evaluator changes

## Decision Rule

Promote only if platform AUC exceeds current best `0.820747`.

## Result

- AUC: `0.820747`
- platform inference time: `295s`
- script inference elapsed: `274.882s`
- predictions: `310000`
- outcome: exact AUC tie with current best; faster runtime, but academic track does not score inference time, so do not treat as an AUC promotion
