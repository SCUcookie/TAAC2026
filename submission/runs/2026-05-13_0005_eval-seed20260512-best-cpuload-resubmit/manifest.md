# Eval Seed 20260512 Best CPU Load Resubmit

## Date

2026-05-13

## Purpose

Use the final successful evaluation slot on a controlled resubmission of the current best evaluator package. This is not a new variant; it is a duplicate submission intended to preserve slot usage discipline while avoiding a lower-probability regression package.

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-resubmit.zip`

## Source Package

Copied byte-for-byte from:

`submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Expected Result

Should match the first successful evaluation within normal platform noise:

- reference AUC: `0.820747`
- reference inference time: `412s`

## Decision Rule

This resubmission does not replace the current best unless the platform returns a higher AUC than `0.820747`.
