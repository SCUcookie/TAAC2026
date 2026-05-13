# 2026-05-11 1635 Training 2GPU Seed 20260511

## Purpose

Start a new 2-GPU training job while evaluation is unavailable for about 18 hours.

This is not an exact rerun. It keeps the best-known archived architecture and training recipe, but changes the seed from the default `42` to `20260511` so the job can explore training variance without changing the model/data recipe.

## Upload Directory

`submission/platform_uploads/2026-05-11_training_2gpu-seed20260511/`

Upload the Python files and `run.sh` from this directory.

## Controlled Change

- Added `--seed 20260511` to `run.sh`.

Everything else remains aligned with the current best 2-GPU recipe:

- `--high_cardinality_mode hash`
- full sequence defaults from `train.py`: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- `--batch_size 256` default
- `--num_workers 8`
- `--use_ema`
- `--random_id_mask_prob 0.05`
- `--rank_mixer_mode learned`
- `--num_queries 2`
- wide dense tokens, signlog dense transform, xdomain features, inter/continuous time, target-aware attention, cross-domain attention, SwiGLU, RMS norm

## Evaluation Rule

When evaluation is available again, compare this checkpoint first with:

`submission/platform_uploads/2026-05-11_eval_archive-fast-single-pass.zip`

Baseline to beat:

- AUC `0.818178`
- inference time about `400s`

Do not use the multiview evaluator as the first comparison, because it scored `0.817438` and took about `1000s`.
