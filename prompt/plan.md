# Current Plan

## Current Best

- Best AUC: `0.820747`
- Best inference time: `412s`
- Best evaluator package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`
- Best evaluator snapshot: `submission/runs/2026-05-13_0001_eval-seed20260512-best-cpuload/eval/`
- Best checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Checkpoint family rule: direct CUDA checkpoint load is not stable here; CPU-staged checkpoint loading is the default safe path

## 2026-05-13 Summary

Three evaluation attempts were completed around the same `20260512` seed checkpoint family.

### Evaluation 1: Stable Single-Pass CPU-Load

Completed and promoted:

- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`
- result: `AUC 0.820747`
- inference time: `412s`

This is the current promoted best pair.

### Evaluation 2: Tail Blend Retry

The first tail-blend attempt failed and does not count as a successful daily submission.

The repaired retry finished but regressed:

- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_tailblend-w005-bs128-cpuload.zip`
- result: `AUC 0.820481`
- inference time: `882s`

Decision: do not promote. Tail-view blend hurt both AUC and latency for this checkpoint.

### Evaluation 3: Recent-Crop TTA

The last useful evaluation slot was used on a new recent-history crop TTA variant:

- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_recent-crop-tta-cpuload.zip`
- result: `AUC 0.820742`
- inference time: `430s`

Decision: very close, but still below the current best `0.820747`. Do not promote.

### Previous Training Package Prepared

- training package: `submission/platform_uploads/2026-05-13_training_abs-time-context-seed20260513/`
- run snapshot: `submission/runs/2026-05-13_0004_training-abs-time-context-seed20260513/train/`
- matching future eval snapshot: `submission/runs/2026-05-13_0004_training-abs-time-context-seed20260513/eval/`

Controlled change versus the proven `20260512` recipe:

- seed `20260512 -> 20260513`
- add bounded absolute timestamp context from row `timestamp`
- derive compact UTC+8 hour/weekday features
- inject them as one extra model token

Implementation status:

- dataset / model / trainer / train wiring completed
- eval compatibility for the future checkpoint completed
- syntax and import checks passed locally
- 2026-05-14 decision: deprioritize this package for now. It changes model shape and has no positive platform signal.

## Current Next Task

The 2026-05-14 recent-crop-regularized training completed successfully, but it should not be evaluated.

Best local point from `temp.md`:

- epoch: `6`
- total step: `21744`
- validation AUC: `0.866055225247`
- validation logloss: `0.220998421311`
- final state: early stopping at epoch `11`, training exit code `0`

Decision: reject for platform evaluation. The best local AUC is below the current winning `20260512` checkpoint family local signal `0.866320891869`, and the curve decays sharply after epoch 7. Do not spend a scarce evaluation chance on this checkpoint.

Today's next action is eval-first, not training-first. Do not submit the prepared 2026-05-13 absolute-time-context training package as the next task.

Reason: that package was only the remaining prepared unused training-side artifact, but it changes model shape and has no platform signal. If today's priority is three evaluation processes, the 2026-05-13 training directory is not the correct next action.

Use the three evaluation opportunities as a result-dependent sequence. The recent-crop-regularized checkpoint is a low-confidence candidate because its best local AUC is below the current best local signal; evaluate it only if the goal is to get direct platform feedback on the new training branch despite the local reject signal.

## Important Guardrails

- Do not submit more evaluation variants on the old `20260512` checkpoint family.
- Do not use older direct-CUDA-load evaluators for this checkpoint family.
- Do not evaluate weak local checkpoints just to use slots.
- Keep the current promoted best unchanged unless a future platform result exceeds `0.820747`.
- The competition end date is `2026-05-23`, so remaining work should stay narrow and leaderboard-driven.

## 2026-05-14 Update

### Evaluation Results

Evaluation `2026-05-14_0007_eval-seed20260512-recent-crop-tta-w004-cpuload` completed:

- package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`
- result: `AUC 0.820747`
- platform inference time: `295s`
- predictions: `310000`
- decision: exact AUC tie with current best. It is faster, but academic track does not score inference time, so keep the promoted best unchanged under the strict AUC-improvement rule.

Evaluation `2026-05-14_0008_eval-seed20260512-recent-crop-tta-long256-w004-cpuload` completed:

- package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-long256-w004-cpuload.zip`
- result: `AUC 0.820742`
- platform inference time: `416s`
- predictions: `310000`
- decision: same AUC as yesterday's recent-crop TTA and below current best. Do not promote.

Postmortem: the second 2026-05-14 eval was too weak a hypothesis for a scarce platform slot. `seq_a` and `seq_b` already cap at `256`, so `long256` only changed the long domains, and the `0.04` blend was unlikely to move rank enough after the `0.08` variant was already flat. Stop submitting eval-only variants on the old `20260512` checkpoint.

### Training Work

Training package `2026-05-14_0009_training-recent-crop-regularized-seed20260514` prepared:

- package: `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`
- snapshot: `submission/runs/2026-05-14_0009_training-recent-crop-regularized-seed20260514/train/`
- explanation: `docs/2026-05-14_training_recent_crop_regularization.md`
- change: training-only stochastic recent-history crop regularization
- defaults: `train_recent_crop_prob=0.35`, `train_recent_crop_min_ratio=0.60`, seed `20260514`
- evaluator compatibility: unchanged model architecture, so use stable CPU-staged single-pass evaluator for any strong checkpoint

Validation and import checks passed locally before packaging:

- AST syntax check passed for `dataset.py`, `model.py`, `trainer.py`, `train.py`
- import smoke check passed with `PYTHONDONTWRITEBYTECODE=1`
- upload directory matches the run snapshot
- `model.py` and `trainer.py` are hash-identical to the proven `20260512` base

### End-of-Day Decision

The current best remains unchanged:

- AUC: `0.820747`
- package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`

Tomorrow's action is conditional on the result of the recent-crop-regularized training. Do not evaluate it unless local validation is competitive; do not spend platform evaluation slots on weak checkpoints or old-checkpoint inference variants.

## 2026-05-15 Update

### Recent-Crop-Regularized Training Result

Training `2026-05-14_0009_training-recent-crop-regularized-seed20260514` finished:

- log: `temp.md`
- best checkpoint step: `21744`
- best epoch: `6`
- best validation AUC: `0.866055225247`
- best validation logloss: `0.220998421311`
- epoch 7 AUC: `0.866049268051`
- epoch 8 AUC: `0.864430103406`
- epoch 11 AUC: `0.852247892833`
- outcome: early stopped at epoch `11`, exit code `0`

Decision: do not evaluate. This is weaker than the current best local signal from the `20260512` checkpoint family and shows overfitting/instability after the best point.

### Today Submission Plan

Run three evaluation processes before any new training job. Update the later eval choice after each platform result.

1. Eval 1: submit the safest available current-best evaluation path as the control/anchor. Prefer the current promoted package `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`; if operational speed matters for turnaround, use the tied faster package `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`.
2. Eval 2: submit the 2026-05-14 recent-crop-regularized best checkpoint only if you deliberately want direct platform feedback on this training branch. Treat it as exploratory because local AUC `0.866055225247` is below the current best local signal `0.866320891869`.
3. Eval 3: choose after Eval 1 and Eval 2 return. If Eval 2 beats or ties `0.820747`, spend Eval 3 on the same branch's best conservative evaluator/confirmation. If Eval 2 is below `0.820747`, stop evaluating that branch and use Eval 3 only for a different prepared candidate with stronger local or platform evidence.

Training job after evaluations:

- Do not use `submission/platform_uploads/2026-05-13_training_abs-time-context-seed20260513/` as the default next action.
- Submit a training job only after the three eval results indicate what is worth training next.
- If no eval improves the leaderboard, the next training should be a new controlled variant based on the proven `20260512` architecture, not the older absolute-time package by default.

Promotion rule remains strict: promote only if platform AUC exceeds `0.820747`.

## Records To Maintain

Every meaningful run should stay linked across:

- `submission/runs/<run_id>/`
- `submission/platform_uploads/`
- `feedbacks/records/<run_id>.md`
- `feedbacks/platform_runs/<run_id>.md`

Record at minimum:

- package path
- checkpoint path
- local validation AUC/logloss when available
- platform AUC
- inference time
- prediction count
- failure / OOM / import issues
- promotion decision against the current best
