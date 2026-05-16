# Eval Seed 20260512 Recent Crop TTA CPU Load

## Date

2026-05-13

## Purpose

Use the final evaluation slot on a genuinely new inference variant with plausible AUC upside: blend the stable full-history prediction with a small recent-history crop view, while preserving the proven CPU-staged checkpoint load path.

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_recent-crop-tta-cpuload.zip`

## Base Package

Copied from:

`submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Changes

- keep CPU-staged checkpoint loading
- keep the same model weights and schema
- add one extra recency-cropped forward pass
- blend logits with conservative weight `0.08`
- crop lengths:
  - `seq_a:128`
  - `seq_b:128`
  - `seq_c:256`
  - `seq_d:256`
- reduce default eval batch size to `128`
- reduce default eval workers to `4`

## Rationale

Tail-view blending already regressed. A recent-history crop TTA targets a different hypothesis: some users may benefit from stronger emphasis on the most recent sequence evidence without discarding the full-history base view.

## Risk

- slower than the current best single-pass evaluator
- two forwards per batch, so runtime will likely increase materially
- could still regress AUC if the checkpoint already weights recency well enough internally

## Decision Rule

Promote only if platform AUC exceeds current best `0.820747`.

## Result

- AUC: `0.820742`
- inference time: `430s`
- outcome: did not beat the current best single-pass CPU-load evaluator
