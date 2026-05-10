# OOM-Safe Pairwise Training Retry

## Basic Info
- Date: 2026-05-09 15:25
- Run type: train
- Submission directory: `submission/platform_uploads/2026-05-09_training_oom-safe-pairwise.zip`
- Platform round: pending
- Platform score: pending
- Previous best score: `0.813033`

## Training Setup
- Entry script: `run.sh`
- Extra args: none by default
- seq_max_lens: `seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- train_ratio / valid_ratio: default `1.0 / 0.1`
- batch_size / num_workers: `64 / 4`
- model config: RankMixer tokenizer, `user_ns_tokens=5`, `item_ns_tokens=2`, `num_queries=2`, `d_model=64`, `num_hyformer_blocks=2`, `num_heads=4`, `seq_encoder_type=longer`, `seq_top_k=50`
- loss config: `LOSS_TYPE=bce_pairwise`, `PAIRWISE_AUC_WEIGHT=0.05`, `PAIRWISE_MAX_PAIRS=8192`; fallback to focal if the runtime parser lacks `bce_pairwise`
- Environment notes: default launcher reduced memory after the previous first-forward CUDA OOM

## Platform Log Signals
- Finished successfully: pending
- Import / packaging errors: pending
- `PLATFORM_TRAIN_CONFIG`: pending
- Best checkpoint path: pending
- Best validation AUC: pending
- Best validation logloss: pending
- Runtime: pending

## Logs / Errors
- Key log lines: pending platform run
- Error message: pending platform run

## Decision
- Compared with baseline `0.813033`: pending
- Keep / revert config: pending
- Next change: submit this OOM-safe retry, then only start evaluation after a new checkpoint is produced
