# 2026-05-16 Daily Summary

## Final Platform Best

- AUC: `0.820776`
- Package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip`
- Checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Method: old best checkpoint plus `seq_c` domain-ablation logit blend, weight `0.06`

## Evaluation Results

Today refined the `seq_c` ablation weight:

| Weight | Package | AUC | Inference Time | Decision |
| --- | --- | ---: | ---: | --- |
| `0.05` | `2026-05-16_eval_oldbest-domain-ablation-seqc-w005-bs128.zip` | `0.820775` | about `800s` | Improved over `w0.03` |
| `0.06` | `2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip` | `0.820776` | about `960s` | Promoted best |
| `0.07` | `2026-05-16_eval_oldbest-domain-ablation-seqc-w007-bs128.zip` | `0.820775` | about `750s` | Regressed |

Conclusion: the `seq_c` ablation signal is real but saturated. The practical optimum is `0.06`.

## Training Package Prepared

Package:

`submission/platform_uploads/2026-05-16_training_stable-bce-bpr-seqc-drop-seed20260516.zip`

Base:

- stable `20260512` training family
- unchanged architecture
- target-aware attention, cross-domain attention, xdomain dense features, inter-time buckets, continuous time deltas, wide dense tokens, SwiGLU, RMSNorm, EMA

Controlled changes:

- seed `20260516`
- `--loss_type bce`
- `--pairwise_bpr_weight 0.04`
- `--domain_dropout_probs seq_c:0.06`

Rationale:

- Research points toward AUC-oriented objectives and target/time-aware sequence modeling.
- The stable architecture already includes most target/time pieces.
- Platform feedback says mild `seq_c` removal helps, so train with matching `seq_c` robustness instead of more focal-only loss or a larger architecture.

Verification:

- AST parse passed for modified `train.py` and `trainer.py`.
- Upload zip contains exactly seven expected files.

## Next Plan

1. Submit the final training package.
2. Evaluate the resulting checkpoint only if local validation reaches the current gate around `0.86632` or clearly improves key hist buckets.
3. If the training is not locally competitive, implement target-aware domain summary tokens next.
4. After target-aware summaries, add time-decay pooling and then a small late unified interaction block as separate ablations.

## Do Not Do

- Do not spend more eval slots on `w0.05`/`w0.07`.
- Do not evaluate locally weak checkpoints.
- Do not combine target-aware summaries, time-decay pooling, and late unified block in one training run.
