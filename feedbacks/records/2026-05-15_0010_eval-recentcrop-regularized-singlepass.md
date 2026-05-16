# Eval Recent-Crop-Regularized Single-Pass

## Package

`submission/platform_uploads/2026-05-15_eval_recentcrop-regularized-singlepass.zip`

## Purpose

First direct platform evaluation for the 2026-05-14 recent-crop-regularized training branch.

## Checkpoint

Select the training best checkpoint from epoch `6`, step `21744`.

## Decision Rule

Continue this branch only if platform AUC is at least `0.82070`. Promote only if it exceeds `0.820747`.

## Platform Result

- AUC: `0.81824`
- inference time: `350s`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `318.717s`
- score mean: `0.1096762770`
- score std: `0.1860060042`

## Final Decision

Reject this branch for further evaluation today. The result is far below both the continuation threshold `0.82070` and the current best `0.820747`.

Next action: do not submit `2026-05-15_eval_recentcrop-regularized-tta-w003.zip`. Pivot Eval 2 to `2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip` on the current best `20260512` checkpoint.
