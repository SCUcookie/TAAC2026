# Eval Seed 20260512 Recent Crop TTA CPU Load

## Date

2026-05-13

## Submission Directory

`submission/runs/2026-05-13_0006_eval-seed20260512-recent-crop-tta-cpuload/`

## Eval Package

`submission/platform_uploads/2026-05-13_eval_seed20260512_recent-crop-tta-cpuload.zip`

## Base Package

Copied from:

`submission/platform_uploads/2026-05-13_eval_seed20260512_best-cpuload-diagnostics.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

- recent-history crop TTA on logits
- weight `0.08`
- crop lens `seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- CPU-staged load unchanged
- default batch size `128`
- default workers `4`

## Reason

User requires a useful last evaluation rather than a duplicate submission. This is the remaining evaluation-time idea with nontrivial upside that has not already been disproven today.

## Local Verification

- syntax compile: passed
- import check: passed

## Platform Result

- AUC: `0.820742`
- inference time: `430s`
- outcome: essentially tied but still below current best `0.820747`, do not promote
