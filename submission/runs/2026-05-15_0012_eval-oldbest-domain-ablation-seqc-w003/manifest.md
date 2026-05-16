# Eval Old-Best Domain Ablation SeqC W003

## Date

2026-05-15

## Purpose

Fallback innovative Eval 2/3 if the new recent-crop-regularized checkpoint is weak. This tests a different ranking hypothesis from old recent-crop TTA: a small logit blend with a view where long sequence domain `seq_c` is suppressed.

## Eval Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`

## Checkpoint

Use the current best checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes

- CPU-staged checkpoint load
- recent TTA disabled by default
- domain ablation blend weight: `0.03`
- ablated domain: `seq_c`

## Submission Condition

Use only if the new-branch single-pass eval is below `0.82070`, or as Eval 3 if no stronger candidate appears.

Promote only if platform AUC exceeds `0.820747`.

## 2026-05-15 Update

New-branch single-pass Eval 1 returned AUC `0.81824`, so this is now the recommended Eval 2 package.

Select the current best `20260512` checkpoint on the platform:

`global_step18120.layer=2.head=4.hidden=64.best_model`
