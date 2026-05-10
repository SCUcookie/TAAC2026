# Original Baseline BF16 Safe Rollback

## Date

2026-05-09 18:54

## Run ID

`2026-05-09_1854_eval-original-bf16-safe-rollback`

## Code Snapshot

`submission/runs/2026-05-09_1854_eval-original-bf16-safe-rollback/eval/`

## Feedback Folder

`feedbacks/platform_runs/2026-05-09_1854_eval-original-bf16-safe-rollback.md`

## Experiment

Safe rollback after the recency-TTA evaluation failed with CUDA OOM.

## Configuration

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-safe-rollback.zip`
- Model: original baseline checkpoint selected on platform
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_TTA_MODE=none`

## Result

Platform evaluation finished successfully.

- Platform score: `0.813041`
- Previous best platform score: `0.813041`
- Platform inference time: `1293.91s`
- Internal inference time: `1283.838s`
- Predictions: `310000`
- Unique users: `310000`

## Log Notes

The TTA code remains available but is opt-in only. Defaults restored the known successful no-merge BF16 behavior and produced the same `0.813041` platform score.

The latest run was slower than the previous successful no-merge BF16 evaluation: `1293.91s` vs `1118.47s`. Score statistics matched the earlier run, including `mean=0.11910518`, `std=0.15703748`, `p50=0.06005859`, and `p99=0.81640625`.

## Decision

No leaderboard improvement. Keep the earlier no-merge BF16 result as the reference and avoid spending more slots on exact reruns.

## Next Action

Start the 2026-05-10 `New_baseline/` training run and explore a new evaluation method separately. Do not submit the new checkpoint unless it passes the local AUC gate.

## Links

- Submission train snapshot: not involved
- Submission eval snapshot: `submission/runs/2026-05-09_1854_eval-original-bf16-safe-rollback/eval/`
- Full run feedback: `feedbacks/platform_runs/2026-05-09_1854_eval-original-bf16-safe-rollback.md`
- Raw log: `feedbacks/logs_output/eval/2026-05-09_1854_eval-original-bf16-safe-rollback.md`
