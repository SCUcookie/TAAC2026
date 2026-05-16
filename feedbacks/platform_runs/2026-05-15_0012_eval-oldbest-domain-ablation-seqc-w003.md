# Platform Run Record

## Basic Info

- Date: 2026-05-15
- Run type: eval
- Eval package: `submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`
- Checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Previous best AUC: `0.820747`

## Platform Result

- AUC: `0.820768`
- inference time: `763.75s`
- predictions: `310000`
- unique users: `310000`
- script inference elapsed: `744.524s`
- domain ablation: `seq_c`
- domain ablation weight: `0.03`

## Decision

Promote as new best. This improves AUC by `0.000021` over the prior best.

Eval 3 should stay on the same checkpoint and same `seq_c` ablation hypothesis, with a nearby weight refinement to `0.04`.
