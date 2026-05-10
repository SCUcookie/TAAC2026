# Original Baseline BF16 No-Merge Evaluation

## Basic Info
- Date: 2026-05-09 16:47
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-09_eval_original-bf16-no-merge.zip`
- Platform round: completed
- Platform score: `0.813041`
- Previous best score: `0.813033`

## Training Setup
- Entry script: `infer.py`
- Extra args: original baseline checkpoint selected on platform
- seq_max_lens: from original checkpoint train config, expected `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: `2048 / 0`
- model config: original baseline checkpoint, no architecture changes
- loss config: N/A
- Environment notes: BF16 AMP enabled by default; batch merging disabled by default

## Platform Log Signals
- Finished successfully: yes
- Import / packaging errors: no
- `PLATFORM_INFER_CONFIG`: present, `merge_batches=false`, `use_amp=true`, `amp_dtype=bf16`
- `PLATFORM_DATASET_CONFIG`: present, `310000` rows, `1000` row groups
- `PLATFORM_AMP_CONFIG`: present, BF16 supported and enabled
- `PLATFORM_FIRST_BATCH`: present, no score sanitization
- `PLATFORM_INFER_PROGRESS`: reached `310000` predictions at `1113.442s`
- `PLATFORM_INFER_SUMMARY`: present
- Local inference verified: syntax check only
- Platform leaderboard metric: `0.813041`
- Runtime: platform reported `1118.47s`; internal `PLATFORM_INFER_SUMMARY.elapsed_sec=1114.191`
- prediction count: `310000`
- unique user count: `310000`
- score min / max / mean / std: `0.00064468 / 0.9765625 / 0.11910518 / 0.15703748`
- score p01 / p50 / p99: `0.0008049 / 0.06005859 / 0.81640625`

## Logs / Errors
- Key log lines: `PLATFORM_INFER_SUMMARY` emitted `310000` predictions; final progress `rows_per_sec=278.42`, avg timing `forward=0.1306`, `merge=0.0`, `move=0.0061`, `post=0.0002`
- Error message: none

## Decision
- Compared with baseline `0.813033`: improved by `+0.000008`; inference is slower (`1118.47s` vs `314.38s`) but acceptable
- Keep / revert config: keep as a successful reference evaluator
- Next change: prioritize AUC improvement for the last submission
