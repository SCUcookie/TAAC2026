# Eval Old-Best Domain Ablation SeqC W004 BS128

## Date

2026-05-15

## Purpose

Retry-safe version of Eval 3 after `w004` failed with CUDA OOM before inference. It keeps the confirmed `seq_c` domain-ablation refinement but reduces batch/worker pressure.

## Eval Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004-bs128.zip`

## Checkpoint

Use:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes Versus Failed W004

- domain ablation domain: `seq_c`
- domain ablation weight: `0.04`
- forced eval batch size: `128`
- forced eval workers: `2`
- DataLoader prefetch factor: `1`

## Decision Rule

Promote only if AUC exceeds current best `0.820768`. If it regresses or fails, keep `w003` as the promoted best.
