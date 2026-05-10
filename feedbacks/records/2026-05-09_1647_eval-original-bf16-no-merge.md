# Original Baseline BF16 No-Merge Evaluation

## Date

2026-05-09 16:47

## Run ID

`2026-05-09_1647_eval-original-bf16-no-merge`

## Code Snapshot

`submission/runs/2026-05-09_1647_eval-original-bf16-no-merge/eval/`

## Feedback Folder

`feedbacks/platform_runs/2026-05-09_1647_eval-original-bf16-no-merge.md`

## Experiment

Retry original baseline evaluation after the BF16 observable run timed out.

## Configuration

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-no-merge.zip`
- Model: original baseline checkpoint selected on platform
- `EVAL_BATCH_SIZE=2048`
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_PROGRESS_EVERY=10`
- Original sequence lengths preserved

## Result

Platform evaluation completed successfully.

## Log Notes

The failed BF16 observable run showed stable BF16 scores but timed out at `237135 / 310000` predictions. The average timing was `merge=9.1777s`, `forward=0.9442s`, `move=0.1489s`, `post=0.0006s`, so this retry disabled default batch merging.

The successful run emitted `PLATFORM_INFER_SUMMARY` with `310000` predictions and `310000` unique users. Internal inference time was `1114.191s`; platform inference time was recorded as `1118.47s`.

Score stats: min `0.00064468`, max `0.9765625`, mean `0.11910518`, std `0.15703748`, p01 `0.0008049`, p50 `0.06005859`, p99 `0.81640625`, with zero NaN/Inf.

## Decision

No-merge BF16 evaluation works and slightly improves score from `0.813033` to `0.813041` (`+0.000008`). It is slower than the baseline evaluation code (`1118.47s` vs `314.38s`), but this is acceptable because it finishes within platform requirements.

## Next Action

Treat this as the current successful reference run. For the last submission today, prioritize a further AUC improvement over runtime.

## Links

- Submission train snapshot: not involved
- Submission eval snapshot: `submission/runs/2026-05-09_1647_eval-original-bf16-no-merge/eval/`
- Full run feedback: `feedbacks/platform_runs/2026-05-09_1647_eval-original-bf16-no-merge.md`
