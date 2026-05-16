# Platform Run Record

## Basic Info

- Date: 2026-05-14 to 2026-05-15
- Run type: train
- Submission directory: `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`
- Platform score: N/A
- Current best platform AUC: `0.820747`

## Training Result

- Finished successfully: yes
- Log source: `temp.md`
- Exit code: `0`
- Early stopping: epoch `11`
- Best checkpoint step: `21744`
- Best epoch: `6`
- Best validation AUC: `0.866055225247`
- Best validation logloss: `0.220998421311`

Validation curve after best:

- epoch 7: AUC `0.866049268051`, logloss `0.221340551972`
- epoch 8: AUC `0.864430103406`, logloss `0.222628399730`
- epoch 9: AUC `0.862442854313`, logloss `0.224798709154`
- epoch 10: AUC `0.858321680808`, logloss `0.228986203671`
- epoch 11: AUC `0.852247892833`, logloss `0.235696673393`

## Decision

Do not spend an evaluation slot on this model. Its best local AUC is weaker than the current winning `20260512` checkpoint family local signal `0.866320891869`, and the curve degrades sharply after epoch 7.

Next action: run today's evaluation processes first. Do not submit `submission/platform_uploads/2026-05-13_training_abs-time-context-seed20260513/` as the default next action; it was only a prepared unused training artifact and is not aligned with the eval-first plan.
