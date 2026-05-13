# Eval Fallback Seed 20260511 Archive Fast CPU Load

## Date

2026-05-12

## Trigger

The first evaluation package failed before scoring:

- package: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-single-pass.zip`
- selected checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- failure: CUDA OOM during checkpoint load
- prediction count: `0`

Key error from `temp.md`:

`CUDA out of memory. Tried to allocate 240.00 MiB...`

## Fallback Package

Submit:

`submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`

## Controlled Change

Evaluator logic is unchanged. Only checkpoint load memory behavior changes:

- `torch.load(..., map_location='cpu')`
- model stays on CPU until after strict state-dict load
- checkpoint object is released before `model.to(cuda)`
- allocator config uses `expandable_segments`

## Expected Outcome

Succeeded.

## Platform Result

- AUC: `0.819544`
- inference time: `481.66s`
- predictions: `310000`
- status: infer finished and score finished

## Interpretation

This beats the previous best AUC `0.818178` by `+0.001366` with acceptable runtime. The CPU-staged checkpoint load should remain the default for this checkpoint family because the direct CUDA load failed with OOM.

The model-output AUC curve shape is almost the same as the previous model, but the peak is slightly higher, consistent with a seed-variance improvement rather than a new behavior regime.
