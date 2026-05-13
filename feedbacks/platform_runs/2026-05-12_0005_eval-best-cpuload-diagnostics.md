# Platform Run Record

## Basic Info
- Date: 2026-05-12
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-12_eval_best-cpuload-diagnostics.zip`
- Platform round: Eval 3 / 3 for 2026-05-12
- Platform score: `0.819544`
- Previous best score: `0.819544`

## Training Setup
- Entry script: `infer.py`
- Extra args: none
- seq_max_lens: loaded from checkpoint `train_config.json`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: loaded from checkpoint `train_config.json`
- model config: completed 2026-05-11 seed checkpoint
- loss config: N/A
- Environment notes: current best single-pass CPU-load evaluator plus diagnostics

## Platform Log Signals
- Finished successfully: yes
- Import / packaging errors: no
- `PLATFORM_TRAIN_CONFIG`: N/A
- `PLATFORM_INFER_CONFIG`: N/A, archived evaluator
- `PLATFORM_DATASET_CONFIG`: N/A, archived evaluator
- `PLATFORM_INFER_SUMMARY`: archived evaluator does not emit structured summary, but log reached `Inference complete: 310000 predictions`
- Best checkpoint path: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Best validation AUC: `0.8658379184807092`
- Best validation logloss: `0.2212848663330078`
- Local inference verified: not run locally, no local eval data in workspace
- Platform leaderboard metric: `0.819544`
- Runtime: official value not provided in message; internal `DIAG_SCORE.elapsed_sec=434.885`, infer-stage wall time about `469.77s`
- prediction count: `310000`
- unique user count: `310000`
- score min / max / mean / std: `0.0006568803` / `0.9793838859` / `0.1088370721` / `0.1478978372`
- score p01 / p50 / p99: `0.0007122836` / `0.0567224044` / `0.8187775314`

## Logs / Errors
- Key log lines: `DIAG_CONFIG`; `DIAG_DATASET`; `DIAG_FIRST_BATCH`; `DIAG_SEQ`; `DIAG_TENSOR`; `DIAG_PROGRESS`; `DIAG_SCORE`; `DIAG_SCORE_HIST`; `Inference complete: 310000 predictions`; `Score Finished`
- Error message: none

## Decision
- Compared with baseline `0.819544`: tied
- Keep / revert config: keep diagnostic CPU-load package as observable equivalent; non-diagnostic CPU-load package remains a simpler fallback
- Next change: submit the 2026-05-12 seed training job if not already submitted, then evaluate its best checkpoint with the CPU-load evaluator when complete
