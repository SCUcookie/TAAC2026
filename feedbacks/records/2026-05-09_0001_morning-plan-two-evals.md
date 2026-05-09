# Morning Plan: Two Evaluations And New Training

## Date

2026-05-09

## Experiment

Morning handoff after yesterday's timeout and today's local/original-evaluator comparison.

## Configuration

- Last failed platform log: `feedbacks/logs_output/log4.md`
- Timed-out platform run: full precision, `EVAL_BATCH_SIZE=1024`, original sequence lengths
- Newly trained model local/original-evaluator accuracy: `0.812668`
- Original baseline model local/original-evaluator accuracy: `0.813033`
- Active code areas: `baseline/training/` and `baseline/evaluation/`

## Result

No new platform result yet today.

## Log Notes

- Yesterday's `log4.md` reached `181192` predictions after about `1738` seconds, then stopped due to inference timeout.
- The first batch and progress logs did not indicate NaN or checkpoint loading failure.
- The bottleneck is inference throughput with full precision and long sequence lengths.
- The original baseline remains slightly stronger than the newly trained model under the same original evaluation code.

## Decision

Today should launch the next multi-hour training run and the two evaluation submissions concurrently. The two evaluation submissions are expected to finish before the new training run completes.

## Next Action

1. Start the next training run with one controlled change.
2. At the same time, evaluate the original baseline model first among evaluation submissions.
3. Submit the newly trained model as the second evaluation process.
4. For platform evaluation, avoid `EVAL_USE_AMP=1`; use a larger full-precision batch if memory permits, or set and record `EVAL_SEQ_LEN_CAP` if timeout risk remains.
