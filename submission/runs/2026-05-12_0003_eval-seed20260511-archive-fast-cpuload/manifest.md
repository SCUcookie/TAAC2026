# Eval Fallback Seed 20260511 Archive Fast CPU Load

## Date

2026-05-12

## Purpose

Fallback for the failed first evaluation. The original archived fast evaluator failed before inference with CUDA OOM while loading `model.pt`.

## Failure Addressed

The failed package loaded the checkpoint with:

`torch.load(ckpt_path, map_location=device)`

On the platform's shared 1-GPU node this OOMed while allocating an extra `240 MiB`, before any prediction batch ran.

## Controlled Fix

Keep the same evaluator behavior and checkpoint, but reduce CUDA load-time pressure:

- set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch
- build the model on CPU
- load `state_dict` with `map_location='cpu'`
- delete the staged checkpoint object and run `gc.collect()`
- move the fully loaded model to CUDA only after CPU loading completes

## Eval Package

`submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`

## Checkpoint

Use the same completed 2026-05-11 seed checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Decision Rule

Compare any successful platform result directly against AUC `0.818178`.

## Result

- AUC: `0.819544`
- inference time: `481.66s`
- predictions: `310000`
- outcome: new best, promote over previous AUC `0.818178`
