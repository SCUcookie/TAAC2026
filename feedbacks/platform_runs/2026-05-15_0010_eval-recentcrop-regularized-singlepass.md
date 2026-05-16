# Platform Run Record

## Basic Info

- Date: 2026-05-15
- Run type: eval
- Eval package: `submission/platform_uploads/2026-05-15_eval_recentcrop-regularized-singlepass.zip`
- Model branch: `2026-05-14_training_recent-crop-regularized-seed20260514`
- Checkpoint: epoch `6`, step `21744`
- Current best before run: `0.820747`

## Platform Result

- AUC: `0.81824`
- inference time: `350s`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `318.717s`
- score range: `0.0005486578` to `0.9826750755`
- score mean / std: `0.1096762770` / `0.1860060042`

## Decision

Do not continue the recent-crop-regularized branch today. The AUC is below the `0.82070` continuation threshold and well below current best `0.820747`.

Eval 2 should pivot to:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`

Use the current best `20260512` checkpoint:

`global_step18120.layer=2.head=4.hidden=64.best_model`
