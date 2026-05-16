# Training Abs Time Context Seed 20260513

## Date

2026-05-13

## Goal

Launch one bounded academic-track training run that uses absolute timestamp context as an additional unified-block token, without changing the proven inference protocol.

## Package

- training: `submission/platform_uploads/2026-05-13_training_abs-time-context-seed20260513/`
- snapshot: `submission/runs/2026-05-13_0004_training-abs-time-context-seed20260513/train/`
- future eval snapshot: `submission/runs/2026-05-13_0004_training-abs-time-context-seed20260513/eval/`

## Controlled Change

Starting from the proven 20260512 seed recipe:

- seed `20260512 -> 20260513`
- add `use_absolute_time_context=true`
- derive compact UTC+8 hour/weekday features from row `timestamp`
- inject them as one extra NS token

## Expected Benefit

Preserve the strong current model while giving the unified block direct access to absolute time context that may align better with conversion periodicity.

## Next Action

Pre-training evaluation is now closed for 2026-05-13:

- locked best remains `AUC 0.820747`, `412s`
- recent-crop TTA finished at `0.820742`, `430s`

Submit the training directory, wait for completion, then evaluate only the best resulting checkpoint against current best AUC `0.820747`.
