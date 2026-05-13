# Eval 2 Tail Blend W010 CPU Load

## Date

2026-05-12

## Purpose

Second evaluation task for 2026-05-12. Test a conservative true-tail blend on top of the current best CPU-load evaluator.

## Eval Package

`submission/platform_uploads/2026-05-12_eval_tailblend-w010-cpuload.zip`

## Base Result

Current best:

- AUC: `0.819544`
- inference time: `481.66s`
- evaluator: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

- Keep CPU-staged checkpoint loading.
- Add true-tail sequence view from raw suffix windows.
- Blend logits as `0.90 * prefix + 0.10 * tail`.
- Recompute tail xdomain dense features from tail sequence tensors.
- Keep all model weights and checkpoint selection unchanged.

## Expected Cost

Approximately 2x forward cost versus single-pass CPU-load evaluator, because each batch runs prefix and tail views. Expected runtime should remain materially below the failed multiview target-tail path.

## Decision Rule

- Promote only if platform AUC beats `0.819544`.
- If AUC is lower or runtime is too high, preserve the current best package.
