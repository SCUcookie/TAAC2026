# Eval Seed 20260512 Best CPU Load Diagnostics

## Date

2026-05-13

## Purpose

First evaluation task for 2026-05-13. Evaluate the completed 2026-05-12 seed-variance training run using the proven CPU-staged checkpoint load evaluator.

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Checkpoint

Select the completed platform training checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

Full checkpoint path from `temp.md`:

`/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260512184310_d1fea1a8/95cdb4769e107aeb019e1bc9953f1c03/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/model.pt`

## Local Validation

Best validation point from `temp.md`:

- epoch: `5`
- total step: `18120`
- validation AUC: `0.866320891869`
- validation logloss: `0.221072241664`
- saved at: `2026-05-13 00:23:32`

Later epochs regressed:

- epoch 6 AUC: `0.866101654771`
- epoch 7 AUC: `0.865591405267`
- epoch 8 AUC: `0.864976776646`
- epoch 9 AUC: `0.862606715774`

## Evaluator

Copied unchanged from:

`submission/runs/2026-05-12_0005_eval-best-cpuload-diagnostics/eval/`

This preserves:

- CPU-staged checkpoint load
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- single-pass logits scoring path
- bounded diagnostic logging

## Decision Rule

Compare against current best platform AUC:

- current best AUC: `0.819544`
- current best package: `submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`
- current best checkpoint: 2026-05-11 seed `global_step18120.layer=2.head=4.hidden=64.best_model`

Promote only if platform AUC is greater than `0.819544`.

## Result

- AUC: `0.820747`
- inference time: `412s`
- predictions: `310000`
- unique users: `310000`
- internal diagnostic elapsed time: `384.594s`
- outcome: new best, promote over previous AUC `0.819544`
