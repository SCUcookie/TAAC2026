# Eval Seed 20260512 Tail Blend W005 BS128 CPU Load

## Date

2026-05-13

## Purpose

Fallback package after the second evaluation failed with CUDA OOM. Retry the same small tail-view blend with lower inference memory pressure.

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-bs128-cpuload.zip`

## Checkpoint

Use the current best platform training checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Baseline To Beat

Current best:

- AUC: `0.820747`
- inference time: `412s`
- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Failure Addressed

The previous tailblend-w005 package failed during inference:

- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-cpuload.zip`
- failure: CUDA OOM before completed prediction batches
- log signal: model loaded successfully, then `Eval tail blend: enabled=True, weight=0.0500`
- likely cause: two model forwards per batch at `batch_size=256`

## Controlled Changes

Starting from `2026-05-13_0002_eval-seed20260512-tailblend-w005-cpuload`:

- keep tail blend weight: `0.05`
- cap inference batch size at `128`
- cap DataLoader workers at `4`
- keep CPU-staged checkpoint loading

## Risk Notes

This should reduce activation memory enough to avoid the observed 1 GiB allocation failure. It will likely be slower than the failed package would have been at batch size `256`, and much slower than the current single-pass best.

## Decision Rule

Submit only if using the third daily slot on a repaired tail-blend attempt is acceptable. Promote only if platform AUC is greater than `0.820747`; otherwise keep the first 2026-05-13 single-pass CPU-load result as current best.

## Result

- AUC: `0.820481`
- inference time: `882s`
- outcome: completed successfully but worse than the current best single-pass CPU-load result
