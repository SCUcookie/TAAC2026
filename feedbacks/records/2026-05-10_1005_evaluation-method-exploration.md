# Evaluation Method Exploration

## Date

2026-05-10 10:05

## Run ID

`2026-05-10_1005_evaluation-method-exploration`

## Experiment

Explore a useful evaluation method without wasting a platform slot after the latest safe rollback evaluation tied the best score at `0.813041` but ran slower.

## Current Reference

- Best platform score: `0.813041`
- Earlier no-merge BF16 runtime: `1118.47s`
- Latest safe rollback runtime: `1293.91s`
- Latest internal inference time: `1283.838s`
- Latest score stats: `mean=0.11910518`, `std=0.15703748`, `p50=0.06005859`, `p99=0.81640625`

## Constraints

- Avoid the recent-window TTA path that caused CUDA OOM.
- Avoid exact reruns of BF16 no-merge because they preserve rank order and already tied the best score.
- Any evaluation change must alter ranking, keep scores finite and non-negative, and preserve exactly `310000` unique user predictions.

## Candidate Direction

Use local or offline validation first to compare rank-changing inference variants against the stable BF16 no-merge evaluator. Prefer lightweight variants that do not duplicate the full model forward in memory.

## Submission Gate

Only spend a platform evaluation slot when the method has a clear local advantage or is paired with a new model checkpoint that passes the training AUC gate.
