# Training 2GPU Seed 20260512

## Date

2026-05-12

## Upload Directory

`submission/platform_uploads/2026-05-12_training_2gpu-seed20260512/`

Clean zip package, excluding local bytecode cache:

`submission/platform_uploads/2026-05-12_training_2gpu-seed20260512.zip`

## Source

Copied from the proven 2026-05-11 2-GPU seed package:

`submission/platform_uploads/2026-05-11_training_2gpu-seed20260511/`

## Controlled Changes

- Seed changed from `20260511` to `20260512`.
- Training and validation `tqdm` progress bars removed from `trainer.py`.
- Added `--log_every_n_steps 200` and concise structured progress logs.

## Recipe

Same best-known archived recipe:

- `--high_cardinality_mode hash`
- full sequence defaults: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- `--batch_size 256` default
- `--num_workers 8`
- `--use_ema`
- `--random_id_mask_prob 0.05`
- `--rank_mixer_mode learned`
- `--num_queries 2`
- target-aware attention, cross-domain attention, xdomain dense features, inter/continuous time, wide dense tokens, SwiGLU, RMS norm

## Expected Logging

The platform log should stay compact:

- `TRAIN_START`
- `EPOCH_START`
- `TRAIN_PROGRESS` every 200 train steps
- `VALIDATION_START`
- `VALIDATION_RESULT`
- checkpoint save/removal lines
- early stopping / training complete

## Evaluation Rule

When this training finishes, evaluate the best checkpoint with the CPU-load archived evaluator and compare against current best AUC `0.819544`.
