# Eval Bracket And Stable BPR Training

## Date

2026-05-16

## Primary Eval Slot

Submit first:

`submission/platform_uploads/2026-05-15_eval_oldbest-domain-ablation-seqc-w004-bs128.zip`

Promote only if platform AUC beats current best `0.820768`.

## Prepared Conditional Eval Packages

All use the old best checkpoint family and safe loader settings unless noted:

- `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w005-bs128.zip`
  - `seq_c` domain-ablation logit blend weight `0.05`
  - batch size `128`, workers `2`, prefetch `1`
- `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w0025-bs128.zip`
  - `seq_c` domain-ablation logit blend weight `0.025`
  - batch size `128`, workers `2`, prefetch `1`
- `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w0035-bs128.zip`
  - `seq_c` domain-ablation logit blend weight `0.035`
  - batch size `128`, workers `2`, prefetch `1`
- `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w0015-bs128.zip`
  - `seq_c` domain-ablation logit blend weight `0.015`
  - batch size `128`, workers `2`, prefetch `1`
- `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w004-bs64.zip`
  - OOM retry for weight `0.04`
  - batch size `64`, workers `1`, prefetch `1`

## Decision Rules

- If `w0.04` improves, evaluate `w0.05`.
- If `w0.04` regresses, evaluate `w0.025`.
- If `w0.04` OOMs, evaluate `w0.04` BS64.
- Use the third slot on `w0.035` or `w0.015` depending on direction.
- If none beat `0.820768`, keep `2026-05-15_eval_oldbest-domain-ablation-seqc-w003.zip`.

## Platform Result Update

`w0.05` BS128 completed successfully.

- package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w005-bs128.zip`
- AUC: `0.820775`
- inference time: about `800s`
- script inference elapsed: `773.490s`
- predictions: `310000`
- unique users: `310000`

Decision: promote `w0.05` as the current best. It beats `w0.03` (`0.820768`) by `0.000007`.

Next evaluation task: submit `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip` to test whether the positive direction continues above `0.05`.

`w0.06` BS128 completed successfully.

- package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip`
- AUC: `0.820776`
- inference time: about `960s`
- script inference elapsed: `935.895s`
- predictions: `310000`
- unique users: `310000`

Decision: promote `w0.06` as the current best. It beats `w0.05` (`0.820775`) by `0.000001`; the gain is tiny but still positive.

Final evaluation task: submit `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w007-bs128.zip`. If it does not beat `0.820776`, keep `w0.06` as today's promoted best.

`w0.07` BS128 completed successfully.

- package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w007-bs128.zip`
- AUC: `0.820775`
- inference time: about `750s`
- script inference elapsed: `722.891s`
- predictions: `310000`
- unique users: `310000`

Decision: keep `w0.06` as today's promoted platform best. The final bracket is:

- `w0.05`: `0.820775`
- `w0.06`: `0.820776`
- `w0.07`: `0.820775`

Interpretation: `seq_c` ablation is real but saturated. The best inference-side value is effectively `0.06`, and training should regularize against mild `seq_c` over-reliance rather than increase model size.

## Training Package

Prepared:

`submission/platform_uploads/2026-05-16_training_stable-bce-bpr-seqc-drop-seed20260516.zip`

Based on the stable `2026-05-12_training_2gpu-seed20260512` family.

Controlled changes:

- seed `20260516`
- `--loss_type bce`
- added `--pairwise_bpr_weight 0.04`
- added `--domain_dropout_probs seq_c:0.06`
- architecture unchanged
- EMA remains enabled
- early stopping remains validation-AUC based

Rationale:

- Research notes prioritize target-aware sequence modeling, time-aware features, and AUC-oriented objectives. The stable family already enables target-aware attention, cross-domain attention, inter-time buckets, continuous time deltas, xdomain features, EMA, SwiGLU, and RMSNorm.
- Today's evaluations show the only platform-positive direction is a mild `seq_c` ablation blend with optimum around `0.06`.
- The final training job therefore keeps the architecture stable and adds low-risk ranking/robustness changes: BCE+BPR plus training-time `seq_c` domain dropout at `0.06`.

## Verification

- AST parse passed for all modified Python files.
- Eval zips contain exactly `dataset.py`, `infer.py`, `model.py`, `ns_groups.json`.
- Training zip contains exactly `dataset.py`, `model.py`, `ns_groups.json`, `run.sh`, `train.py`, `trainer.py`, `utils.py`.

## Cleanup Note

Temporary extracted inspection/cache directories were not included in the zips. Windows denied deletion of:

- `submission/platform_uploads/_inspect_w004_bs128/`
- `submission/platform_uploads/2026-05-16_training_stable-bce-bpr-seed20260516/__pycache__/`
