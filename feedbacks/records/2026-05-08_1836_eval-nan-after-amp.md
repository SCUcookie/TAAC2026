# Eval NaN After AMP

## Date

2026-05-08 18:36

## Experiment

Original baseline model platform evaluation after increasing eval batch size and enabling AMP.

## Configuration

- Evaluation code: `baseline/evaluation/infer.py`
- Model: original baseline checkpoint
- Eval batch size in log: `2048`
- AMP: enabled by previous patch

## Result

Inference completed all `310000` predictions, but platform failed because scores contained `NaN`.

## Log Notes

- First batch already showed `NaN` sample scores.
- Progress reached `310000` predictions in about `710` seconds.
- Failure: `Invalid prediction score detected: scores must be finite and non-negative`.
- Cause is likely mixed precision instability in this model on long sequence batches.

## Decision

Disable AMP by default, sanitize non-finite scores defensively, and merge small row-group batches before each model forward.

## Next Action

Resubmit patched evaluation files. Do not set `EVAL_USE_AMP=1` unless a separate smoke test confirms finite scores.
