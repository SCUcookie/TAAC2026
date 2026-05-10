# 2026-05-09_1839_eval-original-recent-tta

## Purpose

Final evaluation submission for 2026-05-09, prioritizing AUC over runtime while still using the original baseline checkpoint.

## Evaluation Entry

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-recent-tta.zip`
- Entrypoint: `infer.py`
- Files: `infer.py`, `model.py`, `dataset.py`, `ns_groups.json`
- Model: original baseline checkpoint selected on the platform

## Default Config

- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_TTA_MODE=recent_blend`
- `EVAL_TTA_WEIGHT=0.12`
- `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- Original full-window prediction is preserved and blended with the recency-focused TTA prediction.

## Reason

The leaderboard metric is ROC AUC, so monotonic calibration cannot improve the score. This variant changes ranking by blending the original full-window logit with a recency-focused cropped-window logit. The successful no-merge BF16 run scored `0.813041`; this is a last-shot attempt to improve ranking further.
