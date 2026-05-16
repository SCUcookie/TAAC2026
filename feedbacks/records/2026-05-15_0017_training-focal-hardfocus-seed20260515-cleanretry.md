# Training Focal Hard-Focus Seed 20260515 Clean Retry

## Package

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry.zip`

Alternative directory:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry/`

## Purpose

Retry the focal hard-focus training after the previous package failed before training due to platform-side file mapping corruption.

## Controlled Changes

- seed: `20260515`
- loss: focal
- `focal_alpha`: `0.25`
- `focal_gamma`: `1.5`
- no model architecture change
- no recent-crop augmentation

## Verification

- zip contains exactly the seven expected files
- `dataset.py` and `run.sh` contents are not swapped
- Python syntax check passed

## Decision Rule

Evaluate only if local validation is competitive with `0.866320891869`.
