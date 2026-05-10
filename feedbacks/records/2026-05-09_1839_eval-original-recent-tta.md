# Original Baseline Recent-TTA Evaluation

## Date

2026-05-09 18:39

## Run ID

`2026-05-09_1839_eval-original-recent-tta`

## Code Snapshot

`submission/runs/2026-05-09_1839_eval-original-recent-tta/eval/`

## Feedback Folder

`feedbacks/platform_runs/2026-05-09_1839_eval-original-recent-tta.md`

## Experiment

Last evaluation submission today. Use the original baseline checkpoint but change the evaluation ranking through conservative recency test-time augmentation.

## Configuration

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-recent-tta.zip`
- Model: original baseline checkpoint selected on platform
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_TTA_MODE=recent_blend`
- `EVAL_TTA_WEIGHT=0.12`
- `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`

## Result

Platform evaluation failed before producing predictions.

## Log Notes

ROC AUC is rank-based, so monotonic score transforms are not useful. This variant blended the original full-window logit with a recency-focused cropped-window logit to change ranking while keeping the original checkpoint and full-window prediction as the dominant signal.

The platform failed immediately with CUDA OOM after dataset/model setup. `PLATFORM_TTA_CONFIG` and `PLATFORM_FIRST_BATCH` were not reached, so the failure is not a score-quality signal.

## Decision

Revert this TTA path. It is too memory-risky for the current platform GPU slice.

## Next Action

Submit a safe rollback archive based on the known successful no-merge BF16 evaluator if another attempt is available.

## Links

- Submission train snapshot: not involved
- Submission eval snapshot: `submission/runs/2026-05-09_1839_eval-original-recent-tta/eval/`
- Full run feedback: `feedbacks/platform_runs/2026-05-09_1839_eval-original-recent-tta.md`
