# Next 2-GPU Training Job

## Date

2026-05-11 16:25

## Current Baseline

- Best checkpoint/eval pair: 2-GPU checkpoint with archived single-pass evaluator.
- Best AUC: `0.818178`
- Best inference time: about `400s`
- Evaluator package for future model selection: `submission/platform_uploads/2026-05-11_eval_archive-fast-single-pass.zip`

## Latest Evaluation Result

The multiview target-tail evaluator finished but did not improve the score:

- AUC: `0.817438`
- Inference time: about `1000s`

Conclusion: avoid multiview evaluator for the primary leaderboard path unless diagnosing a new checkpoint.

## Training Package To Submit

Use:

`submission/platform_uploads/2026-05-11_training_archive-2gpu-recipe.zip`

This package keeps the archived recipe:

- `--high_cardinality_mode hash`
- full sequence defaults from `train.py`: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- `--batch_size 256` default
- `--num_workers 8`
- `--use_ema`
- `--random_id_mask_prob 0.05`
- `--rank_mixer_mode learned`
- `--num_queries 2`
- target-aware attention, cross-domain attention, xdomain dense features, inter/continuous time, wide dense tokens, SwiGLU, RMS norm

## Next Evaluation Rule

When the new 2-GPU training job completes, evaluate it first with:

`submission/platform_uploads/2026-05-11_eval_archive-fast-single-pass.zip`

Compare directly against AUC `0.818178`. Only try experimental evaluators after the new checkpoint beats or closely matches this baseline under the single-pass evaluator.
