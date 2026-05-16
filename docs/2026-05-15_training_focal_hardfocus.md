# 2026-05-15 Training Change: Focal Hard-Focus Objective

## Context

Today's evaluation work showed that inference-side changes are near saturation:

- best previous AUC: `0.820747`
- `seq_c` domain-ablation weight `0.03`: AUC `0.820768`
- `seq_c` domain-ablation weight `0.04` safe retry: AUC `0.820772`
- total gain from eval tuning: only `+0.000025`

The recent-crop-regularized training branch scored AUC `0.81824`, so more crop training is not the right default.

## Package

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515/`

## Controlled Change

Base: promoted `20260512` architecture and training recipe.

Changes:

- switch `--loss_type focal`
- set `--focal_alpha 0.25`
- set `--focal_gamma 1.5`
- keep `--focal_alpha_mode fixed`
- seed `20260515`

Everything else remains aligned with the proven `20260512` recipe.

## Rationale

High-score recommendation solutions emphasize hard-negative or ranking-aligned optimization. The current code already supports focal loss, which is the safest available objective-side change without implementing a new pairwise loss path today.

This aims to improve model ranking signal directly, while preserving checkpoint format and evaluator compatibility.

## Stop Rule

Do not evaluate the resulting checkpoint unless local validation is competitive with `0.866320891869`.
