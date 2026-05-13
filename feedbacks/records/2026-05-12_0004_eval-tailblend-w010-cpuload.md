# Eval 2 Tail Blend W010 CPU Load

## Date

2026-05-12

## Context

Current best platform result:

- AUC `0.819544`
- inference time `481.66s`
- checkpoint `global_step18120.layer=2.head=4.hidden=64.best_model`
- evaluator `submission/platform_uploads/2026-05-12_eval_seed20260511_archive-fast-cpuload.zip`

The previous multiview target-tail evaluator was not reused because it scored `0.817438` and took about `1000s`.

## Upload

Submit:

`submission/platform_uploads/2026-05-12_eval_tailblend-w010-cpuload.zip`

## Change

This package merges:

- CPU-staged checkpoint loading from the successful `0.819544` evaluator
- lightweight true-tail blend from the prior tail-blend evaluator

Default blend:

- prefix weight `0.90`
- tail weight `0.10`

## Expected Signals

- log should include `Eval tail blend: enabled=True, weight=0.1000`
- log should include `Loading checkpoint ... with CPU staging`
- log should reach `Inference complete: 310000 predictions`

## Decision

Platform result:

- AUC `0.819304`
- inference time `641.82s`
- predictions `310000`

This is worse than current best `0.819544` by `-0.000240` and slower by about `160s`. Do not promote. Tail blend at weight `0.10` likely changes ranking in an unhelpful direction for this checkpoint.

Next action: use Eval 3 to keep the current best prediction path and add bounded diagnostic logging for score/data understanding, without changing the submitted scores.
