# Eval Recent-Crop-Regularized TTA W003

## Package

`submission/platform_uploads/2026-05-15_eval_recentcrop-regularized-tta-w003.zip`

## Purpose

Conditional Eval 2 for the new training branch, using a small `0.03` recent-crop TTA blend.

## Decision Rule

Submit only if Eval 1 is at least `0.82070`. Promote only if AUC exceeds `0.820747`.

## 2026-05-15 Update

Eval 1 on the same branch returned AUC `0.81824`, so this package should not be submitted today.

Use Eval 2 for the old-best domain-ablation package instead:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`
