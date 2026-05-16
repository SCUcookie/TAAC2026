# Training Recent Crop Regularized Seed 20260514

## Date

2026-05-14

## Purpose

Move away from exhausted eval-only TTA variants and train the proven architecture with stochastic recent-history regularization.

## Training Package

`submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`

## Base Package

Copied from:

`submission/runs/2026-05-12_0001_training-2gpu-seed20260512/train/`

## Controlled Changes

- add training-only recent-crop sequence augmentation
- add `--train_recent_crop_prob 0.35`
- add `--train_recent_crop_min_ratio 0.60`
- change seed from `20260512` to `20260514`
- keep model architecture, optimizer, loss, EMA, scheduler, and checkpoint format unchanged

## Technical Explanation

See:

`docs/2026-05-14_training_recent_crop_regularization.md`

## Evaluation Rule

Evaluate only if the training best checkpoint is locally competitive. Use the stable CPU-staged single-pass evaluator and promote only if platform AUC exceeds `0.820747`.
