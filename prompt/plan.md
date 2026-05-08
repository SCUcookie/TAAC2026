# Plan

## Immediate Tasks

1. Keep the training and evaluation code split in separate folders.
2. Store experiment notes in `feedbacks/records/` in chronological order.
3. Add each new idea or parameter change to this plan before the next run.
4. Verify the local inference output before any platform submission.

## Experiment Order

1. Run a short tail-validation training pass with `SEQ_LEN=48` and `PAIRWISE_AUC_WEIGHT=0.05`.
2. Evaluate labeled local splits with `taac2026_eval.py` and record AUC, accuracy, logloss, and feedback logs.
3. Compare validation accuracy, runtime, and stability.
4. Promote only the strongest checkpoint for a longer run.

## Active Candidate

- Training launcher: `run_taac2026.sh`
- Main axis: sequence window 48 plus small pairwise AUC loss.
- Validation: held-out tail batches via `VAL_STRATEGY=tail`.
- Local labeled evaluation: `python taac2026_eval.py --experiment_name taac2026-tail-pairwise`

## Update Rule

- If feedback changes the next action, update this file immediately.
- Keep the next step small and explicit.
