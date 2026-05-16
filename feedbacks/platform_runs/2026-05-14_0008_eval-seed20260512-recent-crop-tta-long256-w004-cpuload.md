# Eval Seed 20260512 Recent Crop TTA Long256 W004 CPU Load

## Date

2026-05-14

## Submission Directory

`submission/runs/2026-05-14_0008_eval-seed20260512-recent-crop-tta-long256-w004-cpuload/`

## Eval Package

`submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-long256-w004-cpuload.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

- recent-history crop TTA on logits
- weight `0.04`
- crop lens `seq_a:256,seq_b:256,seq_c:256,seq_d:256`
- CPU-staged load unchanged
- no tail blend, target blend, multiview, or absolute-time evaluator

## Platform Log Checks

- loaded checkpoint with CPU staging
- `recent_tta=True`
- `recent_tta_weight=0.0400`
- `recent_tta_crop_lens={'seq_a': 256, 'seq_b': 256, 'seq_c': 256, 'seq_d': 256}`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `387.113s`

## Platform Result

- AUC: `0.820742`
- inference time: `416s`
- outcome: same AUC as the 2026-05-13 recent-crop TTA run; below current best `0.820747`, do not promote

## Postmortem

This was not a good use of an evaluation slot. The intended difference was too weak: `seq_a` and `seq_b` already cap at `256`, so this variant only changed the crop behavior for `seq_c` and `seq_d` while using a small `0.04` blend. Given yesterday's `0.08` recent-crop result was already essentially flat, this had low expected AUC upside.
