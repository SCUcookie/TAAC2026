# 2026-05-11 1610 Eval Multiview Target Tail

## Purpose

Evaluation-only candidate for the new 2-GPU baseline checkpoint.

Known baseline:

- AUC `0.818178`
- Inference time about `400s`
- Evaluator: archived/original single-pass prefix/full-window evaluator

## Successful-Run Trend

- Full sequence windows beat aggressive truncation.
- Repeating identical deterministic recent passes mostly increases runtime.
- The current 2-GPU model performs best when evaluated with its training-time full sequence lengths.
- Evaluation changes are only worth a slot if they alter ranking through a genuinely different sequence view.

## Method

This evaluator uses three sequence views in one Parquet read:

- Prefix/full view: original training-compatible view.
- Tail view: suffix window using `values[end - use_len:end]`.
- Target-centered view: when candidate item id appears in the primary sequence, choose a window around the latest hit; otherwise fall back to the tail window.

Default logit blend:

- prefix/full: `0.75`
- tail: `0.15`
- target-centered: `0.10`

Environment controls:

- `EVAL_TAIL_BLEND`, default `true`
- `EVAL_TAIL_BLEND_WEIGHT`, default `0.15`
- `EVAL_TARGET_BLEND`, default `true`
- `EVAL_TARGET_BLEND_WEIGHT`, default `0.10`

The evaluator recomputes xdomain dense features for tail and target views so target-hit features match the active sequence view.

## Rationale

The method is based on the project evidence plus common sequential recommendation practice:

- sequence crop/window augmentations are standard in contrastive sequential recommendation;
- self-attentive recommenders rely on both short-term and long-term behavior;
- this model explicitly uses target-aware attention and target-hit xdomain features, so a target-centered view fits the architecture better than blind repeated TTA.

## Package

`submission/platform_uploads/2026-05-11_eval_multiview-target-tail-w015-w010.zip`
