# Training Focal Hard-Focus Seed 20260515

## Date

2026-05-15

## Purpose

Model-side training attempt after today's evaluation refinements only improved AUC by very small margins. This package keeps the promoted `20260512` architecture and changes only the loss objective to focus learning on hard examples.

## Training Package

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515/`

## Base

Copied from:

`submission/runs/2026-05-12_0001_training-2gpu-seed20260512/train/`

## Controlled Changes

- seed: `20260512 -> 20260515`
- loss: `bce -> focal`
- `focal_alpha`: `0.25`
- `focal_gamma`: `1.5`
- `focal_alpha_mode`: `fixed`
- no model architecture change
- no sequence crop augmentation
- evaluator compatibility unchanged

## Why This Instead Of More Eval Tricks

Today's best eval refinement reached AUC `0.820772`, only `+0.000025` over the previous best `0.820747`. The model branch from recent-crop training scored only `0.81824`, so the next training should return to the proven architecture and change the learning objective rather than keep tuning inference.

## After Training

Evaluate only if local validation is competitive with the current best local signal `0.866320891869`. Use the stable CPU-staged single-pass evaluator first.
