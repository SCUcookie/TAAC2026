# Plan

## Immediate Tasks

1. Use the existing successful `baseline/` run as the benchmark.
2. Improve model training code and evaluation strategy to beat score `0.8133`.
3. Make all active code changes under `baseline/training/` and `baseline/evaluation/`.
4. Store experiment notes in `feedbacks/records/` in chronological order.
5. Add each new idea or parameter change to this plan before the next run.
6. Verify the local inference output before any platform submission.

## Current Benchmark

- Source: already-running `baseline/` split.
- Status: one trained model and one successful evaluation result.
- Score to beat: `0.8133`.
- Do not repeat baseline setup unless needed for a controlled comparison.

## Today's Evaluation

- First evaluation should submit the very original baseline model.
- Submitted evaluation code must include `print` feedback logs so platform terminal output captures checkpoint, dataset, prediction count, score distribution, and output-file status.
- Do not let experimental training changes affect this original baseline submission.

## Experiment Order

1. Start from the working baseline code path.
2. Apply one model or evaluation-method change intended to improve beyond `0.8133`.
3. Run a short validation/smoke pass before any long training.
4. Evaluate with the updated evaluation method and record accuracy/score plus logs.
5. Promote only checkpoints that clearly improve against the `0.8133` benchmark.

## Active Candidate

- Training code directory: `baseline/training/`
- Evaluation code directory: `baseline/evaluation/`
- Local labeled evaluation: `python baseline/evaluation/local_eval.py --experiment_name baseline-local-eval`
- Improvement method to explore after first submission: `LOSS_TYPE=bce_pairwise` with sampled pairwise AUC loss.
- Baseline default remains `LOSS_TYPE=bce`; set `LOSS_TYPE=bce_pairwise` only for the new experiment.

## Update Rule

- If feedback changes the next action, update this file immediately.
- Keep the next step small and explicit.
