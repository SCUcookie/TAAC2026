# Evaluation Timeout Error

## Run ID

2026-05-09_1048_launch-training-and-eval1

## Model Side

Model used: original baseline checkpoint

## Failure Summary

The evaluation failed because the inference job timed out before all predictions were produced.

## Evidence

- `PLATFORM_INFER_PROGRESS` reached `181192` predictions.
- The last visible progress line showed about `1738.428` seconds elapsed.
- There was no checkpoint loading error and no NaN sanitization failure in this run.

## Existing Raw Archive

- `feedbacks/logs_output/log4.md`

## Next Step

If resubmitting the original baseline model, reduce inference cost or split the workload so the timeout threshold is no longer hit.
