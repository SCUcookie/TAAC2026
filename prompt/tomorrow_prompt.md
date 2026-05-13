# Tomorrow Startup Prompt

Read `prompt/codex_workflow.md`, then `prompt/plan.md`, then the latest records under `feedbacks/records/`.

## Current Best

- Best AUC: `0.819544`.
- Inference time: `481.66s`.
- Best evaluator: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`.
- Best checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`.
- Best local validation: epoch 5 AUC `0.8658379184807092`, logloss `0.2212848663330078`.
- Direct CUDA checkpoint load failed with OOM; CPU-staged checkpoint load succeeded and should be reused.

## Today's Unfinished Work

There are still 2 evaluation tasks and 1 training task planned.

### Evaluation 2

Completed:

- package `submission/platform_uploads/2026-05-12_eval_tailblend-w010-cpuload.zip`
- AUC `0.819304`
- inference time `641.82s`
- result was worse than current best `0.819544`

Do not promote tail blend weight `0.10`.

### Evaluation 3

Completed with diagnostic confirmation package:

`submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`

Result:

- AUC `0.819544`
- predictions `310000`
- unique users `310000`
- internal diagnostic elapsed time `434.885s`
- infer-stage wall time about `469.77s`

It keeps the current-best scoring path and adds bounded diagnostic logs. Use it as the preferred observable evaluator unless a later platform run beats `0.819544`.

### Training

Submit:

`submission/platform_uploads/2026-05-12_training_2gpu-seed20260512/`

Purpose:

- controlled seed-variance run
- seed `20260512`
- same proven 2-GPU archived recipe
- concise logs with `--log_every_n_steps 200`

When it finishes, evaluate its best checkpoint first with the CPU-load archived evaluator.

## Final Ten-Day Direction

The competition plan is now leaderboard-first:

- AUC is the primary objective.
- Latency is a hard validity constraint.
- Keep one promoted best checkpoint/evaluator pair.
- Use platform eval slots only for high-confidence comparisons.
- Maintain complete run records for every train/eval attempt.

Priority ideas:

1. seed variance and checkpoint selection
2. conservative ranking-changing evaluator variants
3. timestamp and sequence-interaction features
4. partial-data fast validation before full training
5. larger architecture changes only after a quick signal

Carry forward from `temp.md`:

- explicit sequence-feature interaction rules may help
- timestamp-aware train/eval alignment is worth testing
- partial continuous-time data can validate ideas faster
- industrial-track ideas can guide design when rules and latency allow
- do not maximize every suggestion blindly

## Guardrails

- Do not use direct CUDA checkpoint load for this checkpoint family.
- Do not spend slots on deterministic repeated inference.
- Do not use the old multiview target-tail evaluator as the default path.
- Do not overwrite the `0.819544` best pair unless a platform result beats it.
