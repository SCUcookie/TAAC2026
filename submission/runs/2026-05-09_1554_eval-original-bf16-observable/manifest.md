# 2026-05-09_1554_eval-original-bf16-observable

## Purpose

Evaluate the original baseline checkpoint again without changing the model weights, while reducing timeout risk and collecting richer platform logs.

## Evaluation Entry

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-observable.zip`
- Entrypoint: `infer.py`
- Files: `infer.py`, `model.py`, `dataset.py`, `ns_groups.json`
- Model: original baseline checkpoint selected on the platform

## Default Config

- `EVAL_BATCH_SIZE=2048`
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_PROGRESS_EVERY=10`
- Original checkpoint sequence lengths are preserved unless an emergency env override is set.

## Logging

The evaluator prints:

- `PLATFORM_INFER_START`
- `PLATFORM_CHECKPOINT_STATUS`
- `PLATFORM_INFER_CONFIG`
- `PLATFORM_DATASET_CONFIG`
- `PLATFORM_AMP_CONFIG`
- `PLATFORM_FIRST_BATCH`
- `PLATFORM_INFER_PROGRESS`
- `PLATFORM_AMP_FALLBACK` when BF16 is unsupported or produces non-finite scores
- `PLATFORM_SCORE_SANITIZE` if any scores require sanitization
- `PLATFORM_INFER_SUMMARY`

## Reason

The previous full-precision evaluation timed out after about `181192 / 310000` predictions. A prior fp16 AMP attempt completed quickly but produced NaNs. This retry uses BF16 AMP by default and falls back to fp32 if non-finite scores appear.
