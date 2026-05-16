# Training Recent-Crop V2 Low-Variance Seed 20260515

## Package

`submission/platform_uploads/2026-05-15_training_recent-crop-v2-lowvar-seed20260515/`

## Purpose

Lower-variance recent-crop training follow-up with crop probability `0.18` instead of `0.35`.

## Decision Rule

Submit after today's eval results if recent-crop remains plausible or no eval improves. Evaluate only if local validation is competitive with `0.866320891869`.

## 2026-05-15 Decision

Do not submit this as today's primary training job. The recent-crop branch produced platform AUC `0.81824`, so the safer model-side attempt is the focal hard-focus package.
