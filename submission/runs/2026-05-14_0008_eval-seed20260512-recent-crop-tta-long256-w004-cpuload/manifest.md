# Eval Seed 20260512 Recent Crop TTA Long256 W004 CPU Load

## Date

2026-05-14

## Purpose

Test a less aggressive, domain-selective recent-crop TTA variant by preserving short-domain full-sequence behavior and applying a 256-token recent crop uniformly.

## Eval Package

`submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-long256-w004-cpuload.zip`

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
- set crop lengths:
  - `seq_a:256`
  - `seq_b:256`
  - `seq_c:256`
  - `seq_d:256`
- no tail blend, target blend, multiview, or absolute-time-context evaluator changes

## Decision Rule

Promote only if platform AUC exceeds current best `0.820747`.

## Result

- AUC: `0.820742`
- platform inference time: `416s`
- script inference elapsed: `387.113s`
- predictions: `310000`
- outcome: exact AUC match to the 2026-05-13 recent-crop TTA result, with only a small runtime improvement; do not promote
