# Eval Tail Blend On New Baseline

## Date

2026-05-11 15:45

## Goal

Use the new 2-GPU model as the baseline and attempt an evaluation-only AUC improvement over AUC `0.818178`.

## Change

Implemented true raw-sequence tail-window inference:

- normal view remains prefix/full-window truncation;
- tail view uses `values[end - use_len:end]` for sequence side-info and timestamps;
- tail xdomain dense features are recomputed from tail sequence tensors;
- default blend is `0.85 * prefix_logit + 0.15 * tail_logit`;
- controlled by `EVAL_TAIL_BLEND` and `EVAL_TAIL_BLEND_WEIGHT`.

## Submission Candidate

Package:

`submission/platform_uploads/2026-05-11_eval_tail-blend-w015-new-baseline.zip`

Use it against the best 2-GPU checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Decision Logic

Submit this only if spending the remaining evaluation chance on an eval-only AUC attempt is preferred over simply locking in the known `0.818178` result. The expected runtime is around twice the archived single-pass evaluator but should remain within budget.
