# Original Baseline Recent-TTA Evaluation

## Basic Info
- Date: 2026-05-09 18:39
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-09_eval_original-recent-tta.zip`
- Platform round: final eval submission for today
- Platform score: N/A
- Previous best score: `0.813041`

## Training Setup
- Entry script: `infer.py`
- Extra args: original baseline checkpoint selected on platform
- seq_max_lens: original checkpoint full window, expected `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: `2048 / 0`
- model config: original baseline checkpoint, no weight changes
- loss config: N/A
- Environment notes: BF16 AMP enabled, batch merging disabled, recency TTA enabled

## Platform Log Signals
- Finished successfully: no
- Import / packaging errors: no
- `PLATFORM_INFER_CONFIG`: present, `tta_mode=recent_blend`, `tta_weight=0.12`
- `PLATFORM_TTA_CONFIG`: not reached
- `PLATFORM_INFER_SUMMARY`: not reached
- Local inference verified: AST syntax check only
- Platform leaderboard metric: pending
- Runtime: failed immediately after model/dataset setup
- prediction count: 0
- unique user count: 0
- score min / max / mean / std: pending
- score p01 / p50 / p99: pending

## Logs / Errors
- Key log lines: `PLATFORM_INFER_CONFIG` present, then `Inference failed: CUDA error: out of memory`
- Error message: CUDA out of memory

## Decision
- Compared with current best `0.813041`: no score returned
- Keep / revert config: revert TTA path
- Next change: submit a safe rollback archive based on the known successful no-merge BF16 evaluator
