# Current Plan

## Current Best

- Best platform AUC: `0.820776`
- Best evaluator package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip`
- Best checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Checkpoint family: stable `20260512` training family
- Safe inference rule: use CPU-staged checkpoint loading; do not use older direct-CUDA-load evaluators.
- Academic-track note: inference time is not the score, but keep variants under the platform time limit.

## Daily First Job: Three Evaluation Tasks

Every competition day starts by defining and executing the three available evaluation tasks before training or architecture work.

Rules:

- Always reserve the first work block for the day's three evaluation submissions.
- After each platform result, immediately update this plan with AUC, inference time, package path, and promotion decision.
- If an evaluation beats current best, use the remaining slots to bracket/refine around that new best.
- If none beats current best, keep `w0.06` and move to the prepared training/architecture job after all three eval slots are used or explicitly deferred.

Today's evaluation queue:

1. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0055-bs128.zip`
   - refines the left side of the saturated optimum between `w0.05` and `w0.06`
   - result: AUC `0.820775`, inference time `1000s`
   - decision: do not promote; it ties `w0.05` and remains below `w0.06`

2. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w00625-bs128.zip`
   - checks whether the best point is slightly above `w0.06`
   - result: AUC `0.820776`, inference time `850s`
   - decision: AUC tie with current best; keep `w0.06` as official promoted best, but `w0.0625` is an operationally acceptable tied evaluator

3. `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0065-bs128.zip`
   - right-side guardrail below the already-regressed `w0.07`
   - optional final bracket check; promote only if AUC > `0.820776`

## 2026-05-16 Summary

Today used all evaluation slots on the only confirmed platform-positive direction: old best checkpoint plus `seq_c` domain-ablation logit blend.

### Evaluation Bracket

Baseline before today:

- `w0.03`: AUC `0.820768`

Today:

- `w0.05` BS128: AUC `0.820775`, inference time about `800s`
- `w0.06` BS128: AUC `0.820776`, inference time about `960s`
- `w0.07` BS128: AUC `0.820775`, inference time about `750s`

Decision:

- Promote `w0.06`.
- `w0.07` regressed, so the inference-only optimum is shallow and saturated around `0.06`.
- Keep `w0.06` unless a future platform result beats `0.820776`.

### Main Training Job Prepared

Final training package:

`submission/platform_uploads/2026-05-16_training_stable-bce-bpr-seqc-drop-seed20260516.zip`

Base:

- stable `2026-05-12_training_2gpu-seed20260512` architecture and data path
- `d_model=64`, `2` HyFormer blocks, `4` heads
- long sequence defaults: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- target-aware attention, cross-domain attention, xdomain dense features, inter-time buckets, continuous time deltas, wide dense tokens, SwiGLU, RMSNorm, EMA

Controlled changes:

- seed `20260516`
- `--loss_type bce`
- `--pairwise_bpr_weight 0.04`
- `--domain_dropout_probs seq_c:0.06`
- architecture unchanged

Rationale:

- Research notes prioritize AUC-oriented ranking objectives, target-aware interest extraction, and time-aware sequence modeling.
- The stable family already contains the target/time/cross-domain parts.
- Today's platform feedback shows mild `seq_c` removal improves ranking, with optimum around `0.06`.
- Training should therefore regularize `seq_c` reliance and optimize ranking, rather than jump to a larger architecture today.

Verification:

- AST parse passed for modified `train.py` and `trainer.py`.
- Training zip contains exactly `dataset.py`, `model.py`, `ns_groups.json`, `run.sh`, `train.py`, `trainer.py`, `utils.py`.

## Next Immediate Actions

1. Submit the three daily evaluation tasks listed above, updating this file after each result.

2. Then submit the prepared absolute-time plus target-domain-summary training package:

   `submission/platform_uploads/2026-05-17_training_abs-time-target-summary-seed20260517.zip`

3. The 2026-05-16 training checkpoint diagnostic eval was reviewed:

   - best local validation AUC from `temp.md`: `0.866052` at epoch 6
   - gate remains `0.86632`
   - platform diagnostic AUC: about `0.816`
   - decision: reject this trained branch; do not spend more evaluation slots on it

4. Historical package already submitted/finished:

   `submission/platform_uploads/2026-05-16_training_stable-bce-bpr-seqc-drop-seed20260516.zip`

5. When new training finishes, inspect local validation:

   - promote for evaluation only if best validation AUC reaches at least `0.86632`, or clearly beats the focal clean retry and improves hist buckets 1 and 2.
   - prefer epoch 4-6 checkpoints if the curve peaks early.
   - reject if AUC improves only after logloss improves while ranking decays.

6. If locally competitive, prepare evaluator from the promoted `w0.06` inference path:

   - same old-best evaluator logic
   - `seq_c` domain-ablation weight `0.06`
   - safe batch size `128`, workers `2`, prefetch `1`

7. If not locally competitive, do not spend an evaluation slot. Move to the next architecture work item below.

## Next Architecture Work

Priority order for the next implementation cycle:

1. Target-aware domain summary tokens.
   - Candidate item attends to each domain sequence.
   - Add four target-interest tokens into the RankMixer/prediction path.
   - Keep model size stable.

2. Time-decay pooling and session-gap features.
   - Reuse existing `time_delta`, `inter_time_delta`, and bucket inputs.
   - Add domain-level decay summaries without changing inference input format.

3. Small late unified interaction block.
   - Tokens: item target, NS tokens, final query tokens, domain summaries.
   - Keep token count bounded; no full OneTrans rewrite yet.

4. Controlled ablations only.
   - baseline stable package
   - +BCE/BPR
   - +BCE/BPR + `seq_c` dropout
   - +target-aware summaries
   - +time-decay summaries

## Guardrails

- Do not evaluate new training checkpoints below the local gate.
- Do not submit more old-checkpoint inference-only `seq_c` weights unless there is a reason to refine around `0.055` or `0.065`; the gain is now tiny.
- Do not evaluate yesterday's focal clean retry directly; its best local AUC was below the gate.
- Do not make a large architecture jump in the same job as objective/data-robustness changes.
- Keep every platform result linked in `feedbacks/records/` and `submission/platform_uploads/`.

## Key Records

- Main day record: `feedbacks/records/2026-05-16_0001_eval-bracket-and-stable-bpr-training.md`
- Final daily summary: `docs/2026-05-16_daily_summary.md`
