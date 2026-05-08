# First Baseline Evaluation And Pairwise Plan

## Date

2026-05-08

## Experiment

Prepare today's first evaluation with the original baseline model, while preparing a separate method to explore improvement.

## Configuration

- Submission model: original baseline checkpoint.
- Submission evaluation code: `baseline/evaluation/infer.py`.
- Current benchmark: `0.8133`.
- New exploration method: `LOSS_TYPE=bce_pairwise`, BCE plus sampled pairwise AUC loss.

## Result

No new platform score yet in this record.

## Log Notes

- `baseline/evaluation/infer.py` now prints platform feedback events for start, checkpoint files, dataset shape, model readiness, first batch samples, and final output summary.
- The original baseline default remains `LOSS_TYPE=bce`.
- The pairwise AUC method is opt-in only and should be trained after the first baseline submission.

## Decision

Submit the original baseline first. Use its printed logs as feedback for any packaging or data-shape issues.

## Next Action

After the first evaluation result returns, train the `bce_pairwise` candidate and compare against `0.8133`.
