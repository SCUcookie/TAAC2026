# 2026-05-14 Training Change: Recent-Crop Sequence Regularization

## Decision Context

The 2026-05-14 evaluation-only variants did not improve leaderboard AUC. The first smaller-blend recent-crop TTA tied the best AUC exactly, and the second long-domain crop variant returned the same AUC as the previous day. Those results show that post-hoc inference blending on the locked `20260512` checkpoint is exhausted.

This change moves the next attempt back into training. It keeps the proven `20260512` architecture and evaluator compatibility, but adds a training-only regularization path so the model learns to be robust to partial recent histories.

## Package

- Training snapshot: `submission/runs/2026-05-14_0009_training-recent-crop-regularized-seed20260514/train/`
- Platform upload directory: `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`
- Base snapshot: `submission/runs/2026-05-12_0001_training-2gpu-seed20260512/train/`

## Files Changed

- `dataset.py`
  - Adds `train_recent_crop_prob`.
  - Adds `train_recent_crop_min_ratio`.
  - Applies stochastic recent-suffix sequence cropping only when `is_training=True`.
- `train.py`
  - Adds CLI flags for the two new dataset controls.
  - Passes the controls into `get_pcvr_data`.
  - Persists the flags through the existing `train_config.json` sidecar path.
- `run.sh`
  - Enables the augmentation with conservative defaults:
    - `--train_recent_crop_prob 0.35`
    - `--train_recent_crop_min_ratio 0.60`
  - Changes seed from `20260512` to `20260514`.

No model class, trainer loop, optimizer, loss, checkpoint format, or inference script was changed.

## Technical Behavior

For each training batch and each sequence domain independently:

1. Read the configured maximum length for that domain.
2. With probability `train_recent_crop_prob`, sample a crop length uniformly between:
   - `round(max_len * train_recent_crop_min_ratio)`
   - `max_len`
3. If the domain has timestamp values, rank the events by non-negative time distance from the row timestamp and select the closest sampled number of events.
4. Sort the selected event positions back into original order before writing tensors, so sequence order remains stable.
5. Apply the exact same selected positions to:
   - sequence id/side-info tensors
   - sequence time-bucket tensors
   - continuous time-delta tensors
   - inter-arrival time tensors
6. If no timestamp is available, fall back to the normal uncropped prefix path unless the sampled crop is active, in which case the code uses the same bounded length behavior without changing validation.

Validation datasets are constructed without the new parameters, so validation remains full-history. Platform inference also remains full-history because the evaluator reads the saved `seq_max_lens` and model config but does not use the training-only augmentation flags.

## Why This Is More Defensible Than More Eval TTA

The exhausted eval attempts only changed logits after training. They could not teach the model to handle alternate history windows; they only mixed two already-fixed rankings.

This training change changes the learned representation:

- It exposes the model to multiple recent-history views across epochs.
- It reduces dependence on exact long-history coverage.
- It aligns with the observed fact that recent-crop TTA was very close to best but not enough as a post-hoc blend.
- It keeps validation and inference on the official full-history input, so any improvement must come from better generalization rather than an inference-time trick.

## Expected Benefit

The target upside is small but real: improve ranking robustness for users whose oldest history is noisy or distribution-shifted, without removing the ability to use full histories at validation/inference.

Compared with the failed eval-only jobs, this is the next reasonable bounded attempt because it modifies training signal instead of spending another slot on a near-duplicate evaluator.

## Risks

- If old-history evidence is important for a large share of positives, the augmentation may underfit those patterns.
- The augmentation is stochastic per domain, so local validation may have slightly higher training variance.
- Because validation is full-history, a worse validation AUC is a clear reject signal.

The chosen defaults are intentionally conservative:

- `0.35` probability means most domain views remain full-length.
- `0.60` minimum ratio means even cropped long domains keep at least 60% of configured history.

## Evaluation Plan

1. Submit `submission/platform_uploads/2026-05-14_training_recent-crop-regularized-seed20260514/`.
2. Watch local validation AUC in the training log.
3. Only evaluate the best checkpoint if local validation is competitive with the current best checkpoint family.
4. Use the stable CPU-staged single-pass evaluator, not recent-crop TTA, tail blend, target blend, or multiview.
5. Promote only if platform AUC exceeds `0.820747`.

## Stop Rule

If local validation does not beat or very closely match the existing best local checkpoint, do not spend an evaluation slot. No more eval-only variants should be submitted on the old `20260512` checkpoint family.
