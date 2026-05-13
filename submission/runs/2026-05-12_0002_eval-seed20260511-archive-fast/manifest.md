# Eval 1 Seed 20260511 Archive Fast

## Date

2026-05-12

## Purpose

First evaluation task for 2026-05-12. Evaluate the completed 2026-05-11 seed checkpoint with the archived single-pass evaluator.

## Eval Package

`submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-single-pass.zip`

This is a dated copy of the known best archived single-pass evaluator:

`submission/platform_uploads/2026-05-11_eval_archive-fast-single-pass.zip`

## Checkpoint To Select On Platform

Use the completed 2026-05-11 seed training checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`

From `temp.md`, this checkpoint had:

- epoch: `5`
- local validation AUC: `0.8658379184807092`
- local validation logloss: `0.2212848663330078`

## Baseline For Decision

Compare platform result against:

- AUC `0.818178`
- runtime about `400s`

## Decision Rule

- If platform AUC beats `0.818178`, promote this checkpoint/evaluator pair.
- If platform AUC nearly matches `0.818178`, spend the next slot only on one conservative evaluator variant.
- If platform AUC is clearly lower, keep the known best pair and avoid slower experimental evaluators for this checkpoint.
