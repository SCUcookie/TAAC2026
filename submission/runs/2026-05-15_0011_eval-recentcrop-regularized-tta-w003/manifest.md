# Eval Recent-Crop-Regularized TTA W003

## Date

2026-05-15

## Purpose

Result-dependent Eval 2 for the recent-crop-regularized checkpoint. It tests whether training-time crop robustness combines with a smaller inference-time recent-history logit view.

## Eval Package

`submission/platform_uploads/2026-05-15_eval_recentcrop-regularized-tta-w003.zip`

## Controlled Changes

- CPU-staged checkpoint load
- recent-crop TTA enabled
- default blend weight: `0.03`
- crop lens: `seq_a:128,seq_b:128,seq_c:256,seq_d:256`

## Submission Condition

Submit only if Eval 1 on the same checkpoint is at least `0.82070`.

Promote only if platform AUC exceeds `0.820747`.

## 2026-05-15 Update

Eval 1 returned AUC `0.81824`, below the continuation threshold. Do not submit this package today.

Pivot Eval 2 to:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`
