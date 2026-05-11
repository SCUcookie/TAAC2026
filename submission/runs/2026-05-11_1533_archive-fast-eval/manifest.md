# 2026-05-11 1533 Archive Fast Eval

## Purpose

Use the 2-GPU trained checkpoint with the archived/original single-pass evaluator that produced the best observed result:

- AUC: `0.818178`
- Inference time: about `400s`

## Training Snapshot

Path: `submission/runs/2026-05-11_1533_archive-fast-eval/train/`

The launcher matches the archived recipe:

- `--high_cardinality_mode hash`
- `--seq_encoder_type transformer` through train.py default
- `--seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512` through train.py default
- `--batch_size 256` through train.py default
- `--num_workers 8`
- `--use_ema`
- `--random_id_mask_prob 0.05`
- `--rank_mixer_mode learned`
- `--num_queries 2`
- wide dense tokens, signlog dense transform, xdomain features, inter/continuous time, target-aware attention, cross-domain attention, SwiGLU, RMS norm

## Evaluation Snapshot

Path: `submission/runs/2026-05-11_1533_archive-fast-eval/eval/`

The evaluator is the archived/original single-pass path:

- no recent-blend TTA
- no eval-time seq length cap
- uses checkpoint `train_config.json` sequence lengths
- uses `batch_size` and `num_workers` from `train_config.json`, falling back to `256` and `16`
- writes `predictions.json`

## Checkpoint

Use the current best 2-GPU checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

Full platform path from logs:

`/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model`

## Packages

- Eval package: `submission/platform_uploads/2026-05-11_eval_archive-fast-single-pass.zip`
- Training package: `submission/platform_uploads/2026-05-11_training_archive-2gpu-recipe.zip`
