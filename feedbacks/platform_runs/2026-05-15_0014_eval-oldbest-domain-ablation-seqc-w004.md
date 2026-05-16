# Platform Run Record

## Basic Info

- Date: 2026-05-15
- Run type: eval
- Eval package: `submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004.zip`
- Checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Previous best AUC: `0.820768`

## Platform Result

Failed before inference completed.

Error:

`CUDA out of memory. Tried to allocate 148.00 MiB.`

Log signals:

- model loading reached CPU-staged checkpoint load
- config included `domain_ablation_domain=seq_c`
- config included `domain_ablation_weight=0.0400`
- batch size inherited from checkpoint train config: `256`
- num workers inherited from checkpoint train config: `8`

## Decision

Do not reuse this package. Retry with the safe package:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004-bs128.zip`

If the retry fails or regresses, keep `w003` as the promoted best.
