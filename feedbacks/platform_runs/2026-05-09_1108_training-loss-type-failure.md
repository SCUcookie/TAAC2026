# Platform Run Record

## Basic Info
- Date: 2026-05-09 11:08
- Run type: train
- Submission directory: `submission/platform_uploads/2026-05-09_training_bce_pairwise.zip`
- Platform round: training retry candidate
- Platform score: N/A
- Previous best score: 0.8133

## Training Setup
- Entry script: `baseline/training/run.sh`
- Extra args: `LOSS_TYPE=bce_pairwise`, `PAIRWISE_AUC_WEIGHT=0.05`, `PAIRWISE_MAX_PAIRS=8192`
- seq_max_lens: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: default from launcher
- batch_size / num_workers: `256 / 8`
- model config: `rankmixer` tokenizer, `USER_NS_TOKENS=5`, `ITEM_NS_TOKENS=2`, `NUM_QUERIES=2`, `D_MODEL=64`, `NUM_HEADS=4`, `NUM_HYFORMER_BLOCKS=2`
- loss config: pairwise candidate
- Environment notes: runtime parser only accepted `bce` and `focal`

## Platform Log Signals
- Finished successfully: no
- Import / packaging errors: no
- `PLATFORM_TRAIN_CONFIG`: present
- `PLATFORM_INFER_CONFIG`: N/A
- `PLATFORM_DATASET_CONFIG`: N/A
- `PLATFORM_INFER_SUMMARY`: N/A
- Best checkpoint path: none
- Best validation AUC: N/A
- Best validation logloss: N/A
- Local inference verified: no
- Platform leaderboard metric: N/A
- Runtime: failed before model construction
- prediction count: N/A
- unique user count: N/A
- score min / max / mean / std: N/A
- score p01 / p50 / p99: N/A

## Logs / Errors
- Key log lines: `argument --loss_type: invalid choice: 'bce_pairwise' (choose from 'bce', 'focal')`
- Error message: training parser mismatch in runtime copy of `train.py`

## Decision
- Compared with baseline `0.813033`: no comparison possible
- Keep / revert config: keep the idea, change the launcher fallback
- Next change: submit the `focal` fallback archive
