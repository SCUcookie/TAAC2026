# Eval Seed 20260512 Recent Crop TTA W004 CPU Load

## Date

2026-05-14

## Submission Directory

`submission/runs/2026-05-14_0007_eval-seed20260512-recent-crop-tta-w004-cpuload/`

## Eval Package

`submission/platform_uploads/2026-05-14_eval_seed20260512_recent-crop-tta-w004-cpuload.zip`

## Checkpoint

`global_step18120.layer=2.head=4.hidden=64.best_model`

## Controlled Change

- recent-history crop TTA on logits
- weight `0.04`
- crop lens `seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- CPU-staged load unchanged
- no tail blend, target blend, multiview, or absolute-time evaluator

## Platform Log Checks

- loaded checkpoint with CPU staging
- `recent_tta=True`
- `recent_tta_weight=0.0400`
- `recent_tta_crop_lens={'seq_a': 128, 'seq_b': 128, 'seq_c': 256, 'seq_d': 256}`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `274.882s`

## Platform Result

- AUC: `0.820747`
- inference time: `295s`
- outcome: exact AUC tie with current best; faster than the previous `412s` best, but academic track does not score inference time, so keep the promoted best unchanged under the strict `>` AUC rule
