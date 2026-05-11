# Eval Multiview Target Tail

## Date

2026-05-11 16:10

## Goal

Use the new 2-GPU model as baseline and improve over AUC `0.818178` by evaluation-only inference.

## Successful-Record Summary

- `0.813041`: BF16 no-merge evaluator on original checkpoint, full sequence lengths; valid but slow.
- `0.813094`: recent-blend with long windows on original checkpoint; tiny AUC gain but high runtime.
- `0.80424`: one-GPU new model under strong evaluator; not competitive.
- `0.818008`: 2-GPU model with changed evaluator; slower and slightly worse.
- `0.818178`: 2-GPU model with archived single-pass evaluator; current baseline and best known score.

Principle: preserve full training-compatible sequence signal, avoid deterministic repeats, and only spend extra runtime on views that can change ranking.

## Implemented Evaluation Code

Added a multiview evaluator:

- prefix/full view from the original evaluator;
- true tail view from raw suffix windows;
- target-centered view around the candidate item when present;
- view-specific xdomain dense features;
- default logit blend `0.75 prefix + 0.15 tail + 0.10 target`.

## Package

`submission/platform_uploads/2026-05-11_eval_multiview-target-tail-w015-w010.zip`

## Expected Runtime

Roughly 3x the archived single-pass evaluator because it performs up to three model forwards per batch. This should still be materially below the slow repeat-14 TTA path.

## Platform Result

- AUC: `0.817438`
- Inference time: about `1000s`

## Decision

Do not use this evaluator as the default for model selection. It was slower and worse than the archived single-pass evaluator on the same 2-GPU checkpoint:

- archived single-pass: AUC `0.818178`, about `400s`
- multiview target-tail: AUC `0.817438`, about `1000s`

The next training job should use the archived 2-GPU training recipe, and future checkpoint comparisons should first use the archived single-pass evaluator.
