# Training Abs Time Context Seed 20260513

## Date

2026-05-13

## Purpose

Academic-track bounded training update. Keep the proven 20260512 recipe and add one unified-block-friendly absolute-time context token derived from row timestamp.

## Training Package

`submission/platform_uploads/2026-05-13_training_abs-time-context-seed20260513/`

## Matching Future Eval Snapshot

`submission/runs/2026-05-13_0004_training-abs-time-context-seed20260513/eval/`

Use this future evaluator only for checkpoints trained with the new absolute-time context flag.

## Controlled Changes

- add `--use_absolute_time_context`
- add `--time_context_utc_offset_hours 8`
- derive a 6-dim compact feature from `timestamp`:
  - `sin(hour)`, `cos(hour)`
  - `sin(weekday)`, `cos(weekday)`
  - `is_weekend`
  - `log1p(hour)`
- project this feature into one additional NS token inside `PCVRHyFormer`
- keep the rest of the 20260512 recipe unchanged except seed `20260513`

## Decision Rule

Evaluate the best resulting checkpoint with the single-pass CPU-staged evaluator path adapted for the new flag. Promote only if platform AUC exceeds current best `0.820747`.
