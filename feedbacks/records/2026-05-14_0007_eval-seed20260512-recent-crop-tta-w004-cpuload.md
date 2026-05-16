# Eval Seed 20260512 Recent Crop TTA W004 CPU Load

## Context

This was the first planned 2026-05-14 evaluation on the locked `20260512` best checkpoint. It retuned yesterday's recent-crop TTA from `0.08` to `0.04` while keeping the same crop lengths.

## Result

- package: `submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- platform AUC: `0.820747`
- platform inference time: `295s`
- predictions: `310000`
- decision: exact AUC tie with current best; faster runtime is useful operationally, but not an academic-track scoring improvement

## Takeaway

The smaller recent-crop blend recovered the `0.000005` loss from the `0.08` variant and tied the best score exactly. Do not spend more old-checkpoint evaluation slots unless the next planned `long256` variant is already committed; future promotion still requires AUC greater than `0.820747`.
