# Platform Run Record

## Basic Info
- Date: 2026-05-09 10:48
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-09_eval1_original-baseline_bs2048_fp32.zip`
- Platform round: Eval1
- Platform score: N/A
- Previous best score: 0.8133

## Training Setup
- Entry script: `baseline/evaluation/infer.py`
- Extra args: `EVAL_BATCH_SIZE=2048`, `EVAL_USE_AMP=0`
- seq_max_lens: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: `2048 / 0`
- model config: original baseline checkpoint
- loss config: N/A
- Environment notes: checkpoint sidecars were present

## Platform Log Signals
- Finished successfully: no
- Import / packaging errors: no
- `PLATFORM_TRAIN_CONFIG`: N/A
- `PLATFORM_INFER_CONFIG`: present
- `PLATFORM_DATASET_CONFIG`: present
- `PLATFORM_INFER_SUMMARY`: not reached
- Best checkpoint path: original baseline checkpoint directory
- Best validation AUC: N/A
- Best validation logloss: N/A
- Local inference verified: yes, under the same original evaluator family
- Platform leaderboard metric: N/A
- Runtime: timed out after long inference progress
- prediction count: 181192
- unique user count: 181192
- score min / max / mean / std: N/A
- score p01 / p50 / p99: N/A

## Logs / Errors
- Key log lines: `PLATFORM_INFER_PROGRESS` reached `181192` predictions after about `1738.428` seconds
- Error message: platform timeout during inference

## Decision
- Compared with baseline `0.813033`: no score returned
- Keep / revert config: revert the high-cost eval settings
- Next change: lower inference cost or switch checkpoint family
