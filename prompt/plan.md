# Current Plan

## Current Best

- Best model/evaluator pair: completed 2026-05-11 seed checkpoint evaluated with the archived single-pass evaluator using CPU-staged checkpoint load.
- Best AUC: `0.819544`.
- Best inference time: `481.66s`.
- Best checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`.
- Best evaluator package: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`.
- Best local validation for that checkpoint: epoch 5 AUC `0.8658379184807092`, logloss `0.2212848663330078`; early stopping occurred at epoch 10.
- Direct CUDA checkpoint load failed with OOM. CPU-staged checkpoint loading is now the default for this checkpoint family.

## Today: Remaining Tasks

Two evaluation tasks and one training task remain.

### Evaluation 2

Completed with conservative tail blend weight `0.10`.

Result:

- package: `submission/platform_uploads/2026-05-12_eval_tailblend-w010-cpuload.zip`
- AUC `0.819304`
- inference time `641.82s`
- worse than current best by `-0.000240`

Decision: do not promote. Tail blend hurt this checkpoint and added runtime.

### Evaluation 3

Completed with current-best confirmation plus bounded diagnostics.

Package:

`submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`

Result:

- AUC `0.819544`, tied current best
- predictions `310000`
- unique users `310000`
- internal diagnostic elapsed time `434.885s`
- infer-stage wall time from log timestamps about `469.77s`

This keeps the exact current-best scoring path and adds diagnostic print logs for config, first-batch sequence stats, progress timing, and score distribution. Keep it as the preferred observable evaluator; keep the non-diagnostic CPU-load evaluator as the simpler fallback.

### Training

Submit the prepared training directory:

`submission/platform_uploads/2026-05-12_training_2gpu-seed20260512/`

This is a controlled seed-variance experiment:

- same proven 2-GPU archived recipe
- seed `20260512`
- concise training logs via `--log_every_n_steps 200`

When it finishes, evaluate its best checkpoint first with the CPU-load archived evaluator, not the older direct CUDA-load package.

## Final Ten-Day Strategy

Use AUC as the only leaderboard objective, with latency as a hard constraint. Treat all future ideas as weighted hypotheses, not commands to maximize every component.

Daily operating rule:

- keep a single promoted best pair at all times
- spend platform eval slots only on high-confidence checkpoint/evaluator pairs
- keep one slot available for fallback or confirmation when possible
- run one training job per long wait window, chosen from validated hypotheses

Priority order:

1. seed variance and checkpoint selection
2. conservative evaluator variants that change ranking without multiplying runtime too much
3. time and sequence interaction features
4. fast partial-data training to test feature-interaction ideas
5. larger architecture changes only after a quick validation signal

Ideas from `temp.md` to carry forward:

- AUC is the direct optimization target.
- Explicit sequence-feature interactions may help more than asking the model to discover all interactions unaided.
- Timestamp-aware modeling should be tested, especially train/eval alignment to evaluation-time timestamp patterns.
- Partial continuous-time slices can validate ideas faster, even if final full-data training is required.
- Strong industrial-track ideas can be borrowed where competition rules and latency allow.
- Academic-track flexibility can support ensemble/mechanism innovation, but only if the submission remains valid under latency limits.

Avoid:

- direct CUDA checkpoint loading for this model family
- deterministic repeated inference that does not change ranking
- slow multiview paths that already underperformed
- using evaluation slots for packaging experiments
- treating every future suggestion as a max-weight change

## Records To Maintain

Every meaningful run needs a linked record in:

- `submission/runs/<run_id>/`
- `submission/platform_uploads/`
- `feedbacks/records/<run_id>.md`
- `feedbacks/platform_runs/<run_id>.md` after platform execution

For each run, record:

- package path
- checkpoint path
- local validation AUC/logloss when available
- platform AUC
- inference time
- prediction count
- failure or OOM details
- decision against current best
