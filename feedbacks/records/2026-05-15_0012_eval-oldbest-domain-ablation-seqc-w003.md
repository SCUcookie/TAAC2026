# Eval Old-Best Domain Ablation SeqC W003

## Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`

## Purpose

Fallback innovative evaluation on the old best checkpoint. It tests a small logit blend with `seq_c` suppressed, rather than repeating recent-crop or tail-blend pipelines.

## Decision Rule

Use only if the new branch underperforms or no better Eval 3 candidate exists. Promote only if AUC exceeds `0.820747`.

## 2026-05-15 Update

New-branch Eval 1 returned AUC `0.81824`, so this package is now Eval 2.

Use the current best `20260512` checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Platform Result

- AUC: `0.820768`
- inference time: `763.75s`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `744.524s`

## Decision

Promote as new best. It beats the previous best `0.820747` by `0.000021`.

Use Eval 3 to refine the confirmed-positive `seq_c` domain-ablation idea with weight `0.04`.
