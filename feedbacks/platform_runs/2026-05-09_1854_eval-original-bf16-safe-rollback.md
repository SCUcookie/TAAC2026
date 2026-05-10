# Original Baseline BF16 Safe Rollback

## Basic Info
- Date: 2026-05-09 18:54
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-09_eval_original-bf16-safe-rollback.zip`
- Platform round: 2026-05-09 daily evaluation
- Platform score: `0.813041`
- Previous best score: `0.813041`

## Training Setup
- Entry script: `infer.py`
- Extra args: original baseline checkpoint selected on platform
- seq_max_lens: original checkpoint full window
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: `2048 / 0`
- model config: original baseline checkpoint
- loss config: N/A
- Environment notes: BF16 AMP enabled, batch merging disabled, TTA disabled by default

## Platform Log Signals
- Finished successfully: yes
- Import / packaging errors: none observed
- `PLATFORM_INFER_CONFIG`: BF16 AMP, `merge_batches=false`, `tta_mode=none`, original baseline checkpoint
- `PLATFORM_INFER_SUMMARY`: present; `elapsed_sec=1283.838`
- Platform leaderboard metric: `0.813041`
- Runtime: platform reported `1293.91s`; internal `PLATFORM_INFER_SUMMARY.elapsed_sec=1283.838`
- prediction count: `310000`
- unique user count: `310000`
- score min / max / mean / std: `0.00064468 / 0.9765625 / 0.11910518 / 0.15703748`
- score p01 / p50 / p99: `0.0008049 / 0.06005859 / 0.81640625`

## Logs / Errors
- Key log lines: `PLATFORM_INFER_SUMMARY` emitted `310000` predictions; final progress reached `1000` row-group batches with `rows_per_sec=241.63`; avg timing `forward=0.2488`, `merge=0.0001`, `move=0.0069`, `post=0.0002`
- Error message: none

## Decision
- Compared with current best `0.813041`: exact tie, no AUC improvement
- Keep / revert config: keep only as a correctness fallback; prefer the earlier successful no-merge BF16 run as the runtime reference
- Next change: train `New_baseline/` model and explore a non-OOM evaluation method before any new platform slot
