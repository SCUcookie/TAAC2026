# Tomorrow Startup Prompt

Read these first, in order:

1. `prompt/codex_workflow.md`
2. `prompt/plan.md`
3. latest `temp.md`
4. latest files under `feedbacks/records/` and `feedbacks/platform_runs/`
5. `docs/2026-05-14_training_recent_crop_regularization.md`

## Current Best

- Best AUC: `0.820747`
- Best evaluator package: `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`
- Best evaluator snapshot: `submission/runs/2026-05-13_0001_eval-seed20260512-best-cpuload/eval/`
- Best checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- CPU-staged checkpoint loading is required for this checkpoint family.

Inference time is not an academic-track scoring metric. Do not promote or choose submissions based on runtime unless AUC is tied and a process decision requires choosing a package.

## 2026-05-14 Summary

Two eval-only jobs on the old `20260512` checkpoint did not improve AUC:

1. `2026-05-14_0007_eval-seed20260512-recent-crop-tta-w004-cpuload`
   - package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`
   - AUC: `0.820747`
   - inference time: `295s`
   - decision: exact AUC tie with current best; no academic-track improvement

2. `2026-05-14_0008_eval-seed20260512-recent-crop-tta-long256-w004-cpuload`
   - package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-long256-w004-cpuload.zip`
   - AUC: `0.820742`
   - inference time: `416s`
   - decision: same AUC as yesterday's recent-crop TTA and below current best

Postmortem: these were too close to already-tested inference perturbations. Do not submit more eval-only variants on the old `20260512` checkpoint family.

## 2026-05-15 Status

The recent-crop-regularized training finished and should not be evaluated:

- training package: `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`
- log: `temp.md`
- best epoch / step: epoch `6`, total step `21744`
- best validation AUC: `0.866055225247`
- best validation logloss: `0.220998421311`
- later curve: decays to AUC `0.852247892833` by epoch `11`
- decision: reject for platform eval because it is below the current best local signal `0.866320891869`

## Today Task

Run the three evaluation processes first. Do not submit the 2026-05-13 absolute-time-context training directory as the next action.

Why not the 2026-05-13 training directory:

- It was only the remaining prepared unused training artifact.
- It changes model shape by adding an absolute-time context token.
- It has no positive platform signal yet.
- It conflicts with the eval-first operating order for today.

## Today's Evaluation Slot Policy

- Eval 1: submit a current-best anchor. Prefer `submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`; use `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip` only if faster turnaround is operationally more useful.
- Eval 2: evaluate the 2026-05-14 recent-crop-regularized best checkpoint only if direct platform feedback on this training branch is worth one slot despite local AUC `0.866055225247` being below the current best local signal `0.866320891869`.
- Eval 3: decide after Eval 1 and Eval 2. If Eval 2 beats or ties `0.820747`, use Eval 3 for confirmation or a conservative same-branch evaluator. If Eval 2 is below `0.820747`, do not spend Eval 3 on that branch.
- Training job: submit only after the three eval results clarify the next direction; do not default to the old 2026-05-13 absolute-time package.

## Guardrails

- No more eval-only variants on the old `20260512` checkpoint.
- Do not evaluate weak local checkpoints just to use slots.
- Do not use direct CUDA checkpoint loading.
- Keep records updated immediately after each platform training/evaluation result.
- Every meaningful run should link `submission/runs/`, `submission/platform_uploads/`, `feedbacks/records/`, and `feedbacks/platform_runs/` when applicable.
