# Eval Old-Best Domain Ablation SeqC W004

## Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004.zip`

## Purpose

Final Eval 3 refinement after `seq_c` domain-ablation weight `0.03` improved platform AUC to `0.820768`.

## Checkpoint

Use:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

- `EVAL_DOMAIN_ABLATION_DOMAIN`: `seq_c`
- `EVAL_DOMAIN_ABLATION_WEIGHT`: `0.04`

## Decision Rule

Promote only if AUC exceeds `0.820768`. If it regresses, keep the `w003` package as the promoted best.

## Platform Result

Failed before inference completed.

Error:

`CUDA out of memory. Tried to allocate 148.00 MiB.`

The package inherited batch size `256` and workers `8`, and the platform GPU was heavily contended. Do not reuse this package.

Retry with:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004-bs128.zip`
