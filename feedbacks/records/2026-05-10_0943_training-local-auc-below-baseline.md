# Training Finished Below Local Baseline

## Date

2026-05-10 09:43

## Run ID

`2026-05-10_0943_training-local-auc-below-baseline`

## Experiment

Review the training process that finished on 2026-05-09 and decide whether it deserves a platform evaluation slot.

## Result

- Finished: yes
- Validation AUC: `0.859`
- Original baseline validation AUC: `0.861`
- Delta vs original baseline: `-0.002`

## Decision

Do not spend a platform evaluation chance on this checkpoint. It fails the local gate because it is below the original baseline model.

## Next Action

Train a new model from `New_baseline/` and only consider evaluation if the resulting checkpoint beats the `0.861` local baseline gate or has another clearly documented reason to override the gate.
