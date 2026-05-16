# Eval Recent-Crop-Regularized Single-Pass

## Date

2026-05-15

## Purpose

Evaluate the best checkpoint from `2026-05-14_training_recent-crop-regularized-seed20260514` with the stable CPU-staged single-pass evaluator. This is the first direct platform signal for the new training branch.

## Eval Package

`submission/platform_uploads/2026-05-15_eval_recentcrop-regularized-singlepass.zip`

## Checkpoint To Select On Platform

Use the best checkpoint from yesterday's training:

- epoch: `6`
- total step: `21744`
- local validation AUC: `0.866055225247`
- local validation logloss: `0.220998421311`

## Decision Rule

- If platform AUC is `< 0.82070`, stop evaluating this branch.
- If platform AUC is `>= 0.82070`, submit the new-branch TTA package as Eval 2.
- Promote only if platform AUC exceeds `0.820747`.
