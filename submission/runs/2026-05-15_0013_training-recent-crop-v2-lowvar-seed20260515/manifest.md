# Training Recent-Crop V2 Low-Variance Seed 20260515

## Date

2026-05-15

## Purpose

Lower-variance follow-up to the 2026-05-14 recent-crop regularization run. The prior `0.35` crop probability produced a weaker local best and a sharp post-epoch-7 decay, so this version keeps the same architecture but reduces augmentation strength.

## Training Package

`submission/platform_uploads/2026-05-15_training_recent-crop-v2-lowvar-seed20260515/`

## Controlled Changes

- base: `2026-05-14_0009_training-recent-crop-regularized-seed20260514`
- `train_recent_crop_prob`: `0.35 -> 0.18`
- `train_recent_crop_min_ratio`: unchanged `0.60`
- seed: `20260515`
- model architecture unchanged
- evaluator compatibility unchanged

## Submission Condition

Submit after today's eval results if the recent-crop branch is not clearly invalidated, or if no evaluation improves and a safer training-side refinement is needed.

Evaluate the resulting checkpoint only if local validation is competitive with current best local signal `0.866320891869`.

## 2026-05-15 Decision

Do not submit this as today's primary training job. The direct platform evaluation of the recent-crop-regularized branch returned AUC `0.81824`, so the crop branch is weak. Use `2026-05-15_0016_training-focal-hardfocus-seed20260515` instead.
