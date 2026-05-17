# Tomorrow Startup Prompt

Read these first, in order:

1. `prompt/codex_workflow.md`
2. `prompt/plan.md`
3. `docs/2026-05-16_daily_summary.md`
4. latest `temp.md`
5. latest files under `feedbacks/records/`

## Current Best

- Platform AUC: `0.820776`
- Package: `submission/platform_uploads/2026-05-16_eval_oldbest-domain-ablation-seqc-w006-bs128.zip`
- Checkpoint: `global_step18120.layer=2.head=4.hidden=64.best_model`
- Inference idea: `seq_c` domain-ablation logit blend, weight `0.06`

## First Job Every Day: Three Evaluation Tasks

Before any training submission or architecture work, define and submit the day's three evaluation tasks.

Today's evaluation priority:

1. Submit `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0055-bs128.zip`.
   - Purpose: refine the left side of the saturated optimum between `w0.05` and `w0.06`.
   - Promote only if AUC beats current best `0.820776`.

2. Submit `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w00625-bs128.zip`.
   - Purpose: check whether the best point is slightly above `w0.06`.
   - Promote only if AUC beats current best `0.820776`.

3. Submit `submission/platform_uploads/2026-05-17_eval_oldbest-domain-ablation-seqc-w0065-bs128.zip`.
   - Purpose: right-side guardrail below the already-regressed `w0.07`.
   - Promote only if AUC beats current best `0.820776`.

Record each returned AUC, inference time, package path, and promotion decision immediately after every submission. Update `prompt/plan.md` after each result before choosing the next task.

## State To Resume

The latest training package is ready:

`submission/platform_uploads/2026-05-17_training_target-domain-summary-seed20260517.zip`

Submit this only after the three daily evaluation tasks are handled or explicitly deferred.

Why this job:

- The 5-16 training job finished cleanly but peaked at local validation AUC `0.866052`, below the `0.86632` evaluation gate.
- The new package keeps the proven stable architecture and adds target-aware domain summary tokens as a controlled architecture ablation.
- It uses seed `20260517`, BCE, and `pairwise_bpr_weight=0.04`, without `seq_c` training dropout.

## After Training Completes

Evaluate only if local validation is competitive:

- target gate: `0.86632` validation AUC
- also check hist buckets 1 and 2
- preserve epoch 4-6 checkpoints if the curve peaks early

If competitive, build evaluation from the current best `w0.06` safe evaluator path. If not competitive, do not spend an evaluation slot.

## Next Build Direction

If the target-domain-summary job is not locally competitive, implement time-decay pooling/session-gap summaries next. Do not start a full HSTU/OneTrans rewrite yet.
