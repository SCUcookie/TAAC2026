# Plan

## Immediate Tasks

1. Use the existing successful `baseline/` run as the benchmark.
2. Improve model training code and evaluation strategy to beat score `0.8133`.
3. Make all active code changes under `baseline/training/` and `baseline/evaluation/`.
4. Store experiment notes in `feedbacks/records/` in chronological order.
5. Add each new idea or parameter change to this plan before the next run.
6. Verify the local inference output before any platform submission.

## Current Benchmark

- Source: already-running `baseline/` split.
- Status: one trained model and one successful evaluation result.
- Score to beat: `0.8133`.
- Do not repeat baseline setup unless needed for a controlled comparison.

## 2026-05-08 Completed Work

- Submitted the very original baseline model for today's evaluation attempts.
- Added `print` feedback logs to `baseline/evaluation/infer.py` so platform output captures start, checkpoint status, dataset shape, model readiness, first batch samples, progress, score sanitization, and output-file status.
- Fixed the evaluation package import failure by making `baseline/evaluation/infer.py` self-contained and exposing `main()`.
- Diagnosed the timeout: evaluation inherited training batch size `256`, requiring about 1211 small forwards on 310000 rows.
- Patched evaluation to use an explicit eval batch size and progress logs.
- Diagnosed the NaN failure: AMP/mixed precision caused non-finite scores.
- Patched evaluation to disable AMP by default, sanitize non-finite scores defensively, and merge small row-group batches before each model forward.
- Added the opt-in `LOSS_TYPE=bce_pairwise` training path for tomorrow's improvement experiment; baseline default remains `LOSS_TYPE=bce`.

## 2026-05-09 Current Status

- Yesterday's last platform evaluation stopped due to inference timeout; see `feedbacks/logs_output/log4.md`.
- That run used full precision with `EVAL_BATCH_SIZE=1024`, original long sequence lengths, and reached only `181192 / 310000` predictions after about `1738` seconds.
- The newly trained model from `baseline/training/` scored `0.812668` when evaluated with the original evaluation code.
- The original baseline model scored `0.813033` with the same original evaluation code, so it remains the stronger of the two already trained models.
- Today's plan is to launch one new multi-hour training run and the two evaluation submissions concurrently. The evaluation submissions should finish before the new training run completes.

## 2026-05-09 Next Actions

1. Start a new training run from `baseline/training/`; use only one meaningful change from the previous training run so the result is interpretable.
2. In parallel with that training run, submit evaluation for the original baseline model first, because it is the stronger known checkpoint under the original evaluator.
3. Submit evaluation for the newly trained model as the second evaluation process, mainly to compare whether the platform score agrees with the local/original-evaluator result.
4. Use the patched evaluator only with a speed setting that can finish inside the platform timeout. Candidate options:
   - prefer `EVAL_BATCH_SIZE=2048` with full precision if memory permits;
   - if still too slow, set `EVAL_SEQ_LEN_CAP` and record the exact cap;
   - do not enable `EVAL_USE_AMP=1` unless a smoke test confirms finite scores.
5. Confirm each platform log includes `PLATFORM_INFER_SUMMARY` and no meaningful `PLATFORM_SCORE_SANITIZE` counts.
6. Record both returned scores and logs while the new training run continues, then use those results to guide the next training/evaluation decision.

## Experiment Order

1. Start from the working baseline code path.
2. Apply one model or evaluation-method change intended to improve beyond `0.8133`.
3. Run a short validation/smoke pass before any long training.
4. Evaluate with the updated evaluation method and record accuracy/score plus logs.
5. Promote only checkpoints that clearly improve against the `0.8133` benchmark.

## Active Candidate

- Training code directory: `baseline/training/`
- Evaluation code directory: `baseline/evaluation/`
- Local labeled evaluation: `python baseline/evaluation/local_eval.py --experiment_name baseline-local-eval`
- Improvement method to explore after first submission: `LOSS_TYPE=bce_pairwise` with sampled pairwise AUC loss.
- Baseline default remains `LOSS_TYPE=bce`; set `LOSS_TYPE=bce_pairwise` only for the new experiment.
- Evaluation safety defaults: full precision, `EVAL_BATCH_SIZE` controls merged inference batch size, `EVAL_USE_AMP=1` should not be used unless a smoke test confirms finite scores.
- Timeout risk: `EVAL_BATCH_SIZE=1024` plus original sequence lengths was too slow on 2026-05-08; use `2048` or a recorded `EVAL_SEQ_LEN_CAP` for platform attempts.

## Update Rule

- If feedback changes the next action, update this file immediately.
- Keep the next step small and explicit.
