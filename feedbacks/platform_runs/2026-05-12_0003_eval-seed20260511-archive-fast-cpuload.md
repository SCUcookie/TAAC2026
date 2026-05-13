# Platform Run Record

## Basic Info
- Date: 2026-05-12
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`
- Platform round: Eval fallback / successful Eval 1 for 2026-05-12
- Platform score: `0.819544`
- Previous best score: `0.818178`

## Training Setup
- Entry script: `infer.py`
- Extra args: none
- seq_max_lens: loaded from checkpoint `train_config.json`, `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: loaded from checkpoint `train_config.json`
- model config: 2-GPU archived recipe, seed `20260511`
- loss config: N/A
- Environment notes: CPU-staged checkpoint load; selected `global_step18120.layer=2.head=4.hidden=64.best_model`

## Platform Log Signals
- Finished successfully: yes
- Import / packaging errors: no
- `PLATFORM_TRAIN_CONFIG`: N/A
- `PLATFORM_INFER_CONFIG`: not emitted by archived evaluator
- `PLATFORM_DATASET_CONFIG`: not emitted by archived evaluator
- `PLATFORM_INFER_SUMMARY`: archived evaluator does not emit structured summary, but log reached `Inference complete: 310000 predictions`
- Best checkpoint path: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Best validation AUC: `0.8658379184807092`
- Best validation logloss: `0.2212848663330078`
- Local inference verified: not run locally, no local eval data in workspace
- Platform leaderboard metric: `0.819544`
- Runtime: `481.66s`
- prediction count: `310000`
- unique user count: not explicitly logged
- score min / max / mean / std: not logged
- score p01 / p50 / p99: not logged

## Logs / Errors
- Key log lines: `Model loaded successfully on cuda`; `Inference complete: 310000 predictions`; `Saved 310000 predictions`; `Infer Finished`; `Score Finished`
- Error message: none

## Decision
- Compared with baseline `0.818178`: improved by `+0.001366`
- Keep / revert config: keep and promote this checkpoint/evaluator pair as new best
- Next change: only spend Eval 2 on a conservative evaluator variant if expected runtime stays within budget; avoid multiview target-tail because it previously underperformed at about `1000s`
