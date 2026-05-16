# Training Focal Hard-Focus Seed 20260515

## Package

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515/`

## Purpose

Model-side training task after eval-side improvements saturated. Keeps the proven architecture and changes only the loss objective.

## Controlled Changes

- base: `submission/runs/2026-05-12_0001_training-2gpu-seed20260512/train/`
- seed: `20260515`
- loss: focal
- `focal_alpha`: `0.25`
- `focal_gamma`: `1.5`
- no recent-crop augmentation
- evaluator compatibility unchanged

## Decision Rule

Evaluate only if local validation is competitive with `0.866320891869`.

## Platform Result

Failed before training started.

Error:

`SyntaxError: invalid syntax` in platform-side `dataset.py`, line 2, where `dataset.py` contained shell script content.

This is a packaging/upload artifact mapping failure, not evidence against focal loss.

Use clean retry:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry.zip`
