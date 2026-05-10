# Evaluation Result Comparison

## Date

2026-05-09 18:20

## Run ID

`2026-05-09_1647_eval-original-bf16-no-merge`

## Experiment

Original baseline checkpoint evaluated with BF16 AMP and no row-group batch merging.

## Result

- Platform score: `0.813041`
- Inference time: `1118.47s`
- Predictions: `310000`
- Unique users: `310000`

## Baseline Comparison

- Baseline evaluation code score: `0.813033`
- Baseline evaluation code inference time: `314.38s`
- Score delta: `+0.000008`
- Time delta: `+804.09s`

## Decision

The no-merge BF16 evaluator is correct and slightly higher-scoring. Runtime is acceptable as long as it meets platform requirements, so prioritize AUC for the last submission.

## Links

- Log: `feedbacks/logs_output/eval/2026-05-09_1647_eval-original-bf16-no-merge.md`
- Platform record: `feedbacks/platform_runs/2026-05-09_1647_eval-original-bf16-no-merge.md`
