# TAAC2026 Tail Pairwise Setup

## Date

2026-05-08

## Experiment

Prepared the next TAAC2026 model candidate and local labeled evaluation path.

## Configuration

- Training: `VAL_STRATEGY=tail`, `SEQ_LEN=48`, `PAIRWISE_AUC_WEIGHT=0.05`, `FOCAL_GAMMA=0.0`
- Evaluation: `python taac2026_eval.py --experiment_name taac2026-tail-pairwise`

## Result

No full training run was launched in this setup step.

## Log Notes

- Tail validation now keeps the final validation batches out of optimization.
- Local evaluation reports AUC, accuracy, logloss, score statistics, and writes a feedback record.

## Decision

Run a short smoke training pass before any platform evaluation.

## Next Action

Train with a small `MAX_TRAIN_BATCHES` value, then run `taac2026_eval.py` on a labeled holdout split.
