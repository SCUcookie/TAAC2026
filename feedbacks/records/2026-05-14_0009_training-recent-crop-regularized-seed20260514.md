# Training Recent Crop Regularized Seed 20260514

## Context

The 2026-05-14 eval-only recent-crop variants failed to improve AUC. This package switches back to training changes and keeps the proven model/evaluator surface compatible.

## Package

- training: `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`
- snapshot: `submission/runs/2026-05-14_0009_training-recent-crop-regularized-seed20260514/train/`
- explanation: `docs/2026-05-14_training_recent_crop_regularization.md`

## Controlled Change

- training-only stochastic recent-crop augmentation
- `train_recent_crop_prob=0.35`
- `train_recent_crop_min_ratio=0.60`
- seed `20260514`
- no model architecture change

## Decision Rule

Submit training only. Evaluate the produced best checkpoint only if local validation is competitive with the current best. Promote only if platform AUC exceeds `0.820747`.

## Platform Training Result

Log source: `temp.md`

- finished successfully: yes
- exit code: `0`
- early stopping: epoch `11`
- best epoch: `6`
- best total step: `21744`
- best validation AUC: `0.866055225247`
- best validation logloss: `0.220998421311`
- epoch 7 AUC: `0.866049268051`
- epoch 8 AUC: `0.864430103406`
- epoch 9 AUC: `0.862442854313`
- epoch 10 AUC: `0.858321680808`
- epoch 11 AUC: `0.852247892833`

## Final Decision

Do not evaluate this checkpoint on the platform. The best local validation AUC is below the current best checkpoint family local signal `0.866320891869`, and the validation curve deteriorates after epoch 7. Keep the current promoted platform best unchanged at AUC `0.820747`.
