# 2026-05-09_1854_eval-original-bf16-safe-rollback

## Purpose

Safe rollback evaluation archive after the recency-TTA attempt failed with CUDA OOM.

## Evaluation Entry

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-safe-rollback.zip`
- Entrypoint: `infer.py`
- Files: `infer.py`, `model.py`, `dataset.py`, `ns_groups.json`
- Model: original baseline checkpoint selected on the platform

## Default Config

- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_TTA_MODE=none`

## Reason

This restores the known successful BF16 no-merge behavior that scored `0.813041` and completed with `310000` predictions. The TTA code remains present but is opt-in only and disabled by default.
