# Eval Seed 20260512 Tail Blend W005 BS128 CPU Load

## Date

2026-05-13

## Submission Directory

`submission/runs/2026-05-13_0003_eval-seed20260512-tailblend-w005-bs128-cpuload/eval/`

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-bs128-cpuload.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Baseline

- current best platform AUC: `0.820747`
- current best inference time: `412s`

## Controlled Change

- tail blend weight: `0.05`
- batch size cap: `128`
- num_workers cap: `4`
- checkpoint loading: CPU-staged

## Platform Result

- AUC: `0.820481`
- inference time: `882s`
- outcome: worse than current best `0.820747`, do not promote
