# Eval Seed 20260512 Recent Crop TTA Long256 W004 CPU Load

## Context

This was the second planned 2026-05-14 evaluation on the locked `20260512` checkpoint. It attempted a domain-selective recent-crop TTA by setting all crop lengths to `256` and blend weight to `0.04`.

## Result

- package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-long256-w004-cpuload.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- platform AUC: `0.820742`
- platform inference time: `416s`
- predictions: `310000`
- decision: do not promote; same AUC as yesterday's recent-crop TTA and below current best `0.820747`

## Takeaway

Do not submit more eval-only variants on the old `20260512` checkpoint. Future evaluation slots should be reserved for materially different trained checkpoints or changes with a concrete offline signal.
