# Eval Seed 20260512 Tail Blend W005 CPU Load

## Date

2026-05-13

## Submission Directory

`submission/runs/2026-05-13_0002_eval-seed20260512-tailblend-w005-cpuload/eval/`

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-cpuload.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Baseline

- current best platform AUC: `0.820747`
- current best inference time: `412s`
- current best evaluator: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Controlled Change

- base evaluator: 2026-05-12 tailblend CPU-load evaluator
- tail blend weight: `0.05`
- checkpoint loading: CPU-staged

## Platform Result

- status: failed
- error: CUDA out of memory
- failure stage: inference, after model load and before completed prediction batches
- log signal: `Eval tail blend: enabled=True, weight=0.0500`
- checkpoint loaded successfully: yes
- decision: do not resubmit this package unchanged; reduce evaluation batch size if retrying tail blend
