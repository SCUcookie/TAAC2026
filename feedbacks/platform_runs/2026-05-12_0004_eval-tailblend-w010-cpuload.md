# Platform Run Record

## Basic Info
- Date: 2026-05-12
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-12_eval_tailblend-w010-cpuload.zip`
- Platform round: Eval 2 / 3 for 2026-05-12
- Platform score: `0.819304`
- Previous best score: `0.819544`

## Training Setup
- Entry script: `infer.py`
- Extra args: none
- seq_max_lens: loaded from checkpoint `train_config.json`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: loaded from checkpoint `train_config.json`
- model config: completed 2026-05-11 seed checkpoint
- loss config: N/A
- Environment notes: CPU-staged checkpoint load, tail blend weight `0.10`

## Platform Log Signals
- Finished successfully: yes
- Import / packaging errors: no
- `PLATFORM_TRAIN_CONFIG`: N/A
- `PLATFORM_INFER_CONFIG`: archived evaluator does not emit this structured field
- `PLATFORM_DATASET_CONFIG`: archived evaluator does not emit this structured field
- `PLATFORM_INFER_SUMMARY`: archived evaluator does not emit structured summary, but log reached `Inference complete: 310000 predictions`
- Best checkpoint path: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Best validation AUC: `0.8658379184807092`
- Best validation logloss: `0.2212848663330078`
- Local inference verified: not run locally, no local eval data in workspace
- Platform leaderboard metric: `0.819304`
- Runtime: `641.82s`
- prediction count: `310000`
- unique user count: pending
- score min / max / mean / std: pending
- score p01 / p50 / p99: pending

## Logs / Errors
- Key log lines: `Eval tail blend: enabled=True, weight=0.1000`; `Loading checkpoint ... with CPU staging`; `Inference complete: 310000 predictions`; `Score Finished`
- Error message: none

## Decision
- Compared with baseline `0.819544`: worse by `-0.000240`
- Keep / revert config: revert to current best CPU-load single-pass evaluator
- Next change: use Eval 3 for diagnostic logging without changing predictions
