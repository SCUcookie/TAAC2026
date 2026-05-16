# Eval Old-Best Domain Ablation SeqC W004

## Date

2026-05-15

## Purpose

Final Eval 3 refinement after `seq_c` domain-ablation weight `0.03` improved the leaderboard to AUC `0.820768`. This package keeps the same confirmed-positive hypothesis and slightly increases the blend weight to `0.04`.

## Eval Package

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004.zip`

## Checkpoint

Use the current best checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes

- CPU-staged checkpoint load
- recent TTA disabled
- ablated domain: `seq_c`
- domain ablation blend weight: `0.04`

## Decision Rule

Promote only if platform AUC exceeds current best `0.820768`.

If this regresses, keep `2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip` as the promoted best package.
