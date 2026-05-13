# Eval Seed 20260512 Tail Blend W005 CPU Load

## Date

2026-05-13

## Purpose

Second evaluation task for 2026-05-13. Test a small tail-view logit blend on the new best 2026-05-12 seed checkpoint.

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-cpuload.zip`

## Checkpoint

Select the current best platform training checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

Full path from evaluation log:

`/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260512184310_d1fea1a8/self/95cdb4769e107aeb019e1bc9953f1c03/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model`

## Baseline To Beat

Current best:

- AUC: `0.820747`
- inference time: `412s`
- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Controlled Change

Copied from:

`submission/runs/2026-05-12_0004_eval-tailblend-w010-cpuload/eval/`

Changed only:

- default tail blend weight: `0.10` -> `0.05`

The evaluator still uses CPU-staged checkpoint loading and the same model/checkpoint construction.

## Risk Notes

Prior 2026-05-12 tail blend at weight `0.10` was worse on the previous seed:

- single-pass AUC: `0.819544`
- tailblend-w010 AUC: `0.819304`
- delta: `-0.000240`

This slot is a controlled eval-only attempt on the stronger 2026-05-12 seed with a smaller blend weight. Expected runtime is slower than single pass because it performs an extra tail forward pass.

## Decision Rule

Promote only if platform AUC is greater than `0.820747`. Otherwise keep the first 2026-05-13 single-pass CPU-load result as current best.

## Result

- status: failed
- failure stage: inference
- error: CUDA out of memory while starting prediction
- log signal: `Eval tail blend: enabled=True, weight=0.0500`
- likely cause: tail blend performs an extra forward pass at batch size `256`, exceeding the platform GPU slice for this checkpoint
