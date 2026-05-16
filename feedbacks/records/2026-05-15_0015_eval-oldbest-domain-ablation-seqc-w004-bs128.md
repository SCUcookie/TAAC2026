# Eval Old-Best Domain Ablation SeqC W004 BS128

## Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004-bs128.zip`

## Purpose

Safe retry for failed Eval 3. The original `w004` package OOMed before completing inference, likely due GPU contention plus inherited batch size `256` and workers `8`.

## Checkpoint

Use:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes

- domain ablation domain: `seq_c`
- domain ablation weight: `0.04`
- forced eval batch size: `128`
- forced eval workers: `2`
- DataLoader prefetch factor: `1`

## Decision Rule

Promote only if AUC exceeds `0.820768`. Otherwise keep `2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`.
