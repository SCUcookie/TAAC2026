# New Baseline Infer Completed Too Slowly

## Date

2026-05-10 11:36

## Run ID

`2026-05-10_1136_new-baseline-infer-too-slow`

## Experiment

Retry evaluation after adding legacy checkpoint compatibility to `New_baseline/model.py` and `New_baseline/infer.py`.

## Result

Inference completed and emitted `PLATFORM_INFER_SUMMARY`, but total runtime was too high for the platform budget.

- Predictions: `310000`
- Unique users: `310000`
- Internal inference time: `1890.286s`
- Final throughput: about `164` rows/sec
- Batches: `1000` row-group forwards
- Score stats matched the original baseline output exactly.

## Diagnosis

The model forward itself averaged about `0.2386s` per row group. The total elapsed time was dominated by CPU-side parquet conversion and per-row-group iteration overhead. The evaluator still performed one forward per row group despite `EVAL_BATCH_SIZE=2048`, because each parquet row group contained only about `310` rows.

## Fix

Update `New_baseline/infer.py` defaults:

- `EVAL_NUM_WORKERS` default changed from `0` to `4`.
- Default `EVAL_SEQ_LEN_CAP=256` to reduce CPU sequence padding/conversion for `seq_c` and `seq_d`.
- Added optional `EVAL_MERGE_BATCHES=1` support and merge timing logs for controlled throughput testing.
- Kept `EVAL_MERGE_BATCHES=0` by default because earlier platform logs showed CPU merging can be expensive.

## Next Action

For the next smoke evaluation, use the patched defaults first. If it is still slow, try `EVAL_SEQ_LEN_CAP=128`; only try `EVAL_MERGE_BATCHES=1` if first-batch/progress logs show merge time is low.
