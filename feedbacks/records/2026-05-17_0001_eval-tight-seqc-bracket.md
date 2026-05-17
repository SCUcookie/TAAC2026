# Eval Tight Seq-C Bracket

## Date

2026-05-17

## Current Best Before This Bracket

- package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip`
- checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- AUC: `0.820776`
- method: old best `20260512` checkpoint plus `seq_c` domain-ablation logit blend, weight `0.06`

## Planned Evaluation Queue

All packages use the same old-best checkpoint family and safe CPU-staged loader.

1. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0055-bs128.zip`
   - `seq_c` ablation weight `0.055`
2. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w00625-bs128.zip`
   - `seq_c` ablation weight `0.0625`
3. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0065-bs128.zip`
   - `seq_c` ablation weight `0.065`

## Result Update

`w0.055` BS128 completed.

- package: `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0055-bs128.zip`
- AUC: `0.820775`
- inference time: `1000s`
- log source: `temp.md`

Decision: do not promote. It is below current best `0.820776` and slower than the promoted `w0.06` run. Continue with `w0.0625` to test whether the optimum is slightly above `0.06`.

`w0.0625` BS128 completed.

- package: `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w00625-bs128.zip`
- AUC: `0.820776`
- inference time: `850s`
- script inference elapsed from log: `826.658s`
- checkpoint confirmed in log: `global_step18120.layer=2.head=4.hidden=64.best_model`
- config confirmed in log: `domain_ablation_domain=seq_c`, `domain_ablation_weight=0.0625`

Decision: do not mark as a strict leaderboard promotion because AUC ties the current best rather than exceeding it. It is an operationally acceptable tied evaluator and ran faster than the earlier `w0.06` platform run, but academic-track promotion still requires AUC improvement. The final `w0.065` check is optional; use it only if spending the third slot on a right-side guardrail is still more valuable than moving to training.
