# Original Baseline BF16 Observable Evaluation

## Date

2026-05-09 15:54

## Run ID

`2026-05-09_1554_eval-original-bf16-observable`

## Code Snapshot

`submission/runs/2026-05-09_1554_eval-original-bf16-observable/eval/`

## Feedback Folder

`feedbacks/platform_runs/2026-05-09_1554_eval-original-bf16-observable.md`

## Experiment

Retry evaluation on the very original model checkpoint while improving runtime and terminal visibility.

## Configuration

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-observable.zip`
- Model: original baseline checkpoint selected on platform
- `EVAL_BATCH_SIZE=2048`
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_PROGRESS_EVERY=10`
- Original sequence lengths preserved by default

## Result

Platform evaluation timed out before `PLATFORM_INFER_SUMMARY`.

## Log Notes

The previous full-precision evaluation timed out after roughly `181192 / 310000` predictions. Earlier fp16 AMP finished much faster but produced NaNs. This evaluator defaulted to BF16 AMP and automatically reran non-finite batches in fp32.

The new log reached `237135 / 310000` predictions after `1962.399s`. BF16 was stable: no `PLATFORM_AMP_FALLBACK` and no score sanitization appeared. The bottleneck was CPU-side batch merging: average `merge=9.1777s` vs `forward=0.9442s`.

## Decision

Keep BF16, but disable default batch merging. The logs show timeout is data/merge bound, not model-forward bound.

## Next Action

Submit `submission/platform_uploads/2026-05-09_eval_original-bf16-no-merge.zip` and select the original baseline checkpoint.

## Links

- Submission train snapshot: not involved
- Submission eval snapshot: `submission/runs/2026-05-09_1554_eval-original-bf16-observable/eval/`
- Full run feedback: `feedbacks/platform_runs/2026-05-09_1554_eval-original-bf16-observable.md`
