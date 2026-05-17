# Training Abs-Time Target Summary

## Date

2026-05-17

## Diagnostic Eval Before Training

The 2026-05-16 trained checkpoint was evaluated as a diagnostic.

- checkpoint: `global_step21744.layer=2.head=4.hidden=64.best_model`
- evaluator style: current safe evaluator with `seq_c` ablation weight `0.06`
- platform AUC: about `0.816`
- log source: `temp.md`

Decision: reject the 2026-05-16 trained branch for further evaluation. Its platform transfer is much worse than the old-best `20260512` checkpoint family.

## Training Package

Submit:

`submission/platform_uploads/2026-05-17_training_abs-time-target-summary-seed20260517.zip`

Directory:

`submission/platform_uploads/2026-05-17_training_abs-time-target-summary-seed20260517/`

## Base

Built from the prepared target-domain-summary package:

`submission/platform_uploads/2026-05-17_training_target-domain-summary-seed20260517/`

This keeps the proven stable backbone:

- RankMixer tokenizer
- target-aware attention
- cross-domain attention
- xdomain dense features
- relative time buckets
- inter-arrival time buckets
- continuous log time deltas
- wide dense tokens
- SwiGLU
- RMSNorm
- EMA
- BCE plus `pairwise_bpr_weight=0.04`

## New Controlled Change

Adds one absolute-time context token:

- derives UTC+8 hour and weekday from row `timestamp`
- adds learned hour and weekday embeddings
- adds sin/cos cyclic encoding for hour and weekday
- appends the absolute-time token to NS context
- also appends it to target tokens used by target-conditioned domain summaries

Rationale: articles and champion-plan notes emphasize explicit absolute periodic time features. The current stable stack already has relative time-delta modeling, so this package adds the missing absolute/cyclic time signal without changing the data format or making a large architecture jump.

## Verification

- AST parse passed for `dataset.py`, `model.py`, `train.py`, `trainer.py`, and `utils.py`.
- Upload zip contains exactly: `dataset.py`, `model.py`, `ns_groups.json`, `run.sh`, `train.py`, `trainer.py`, `utils.py`.

## Evaluation Gate

Evaluate the resulting checkpoint only if local validation is competitive:

- target gate: at least about `0.86632` validation AUC
- also inspect hist buckets 1 and 2
- reject if local validation is below the gate or if the curve decays like the 2026-05-16 branch
