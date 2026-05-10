# Fast Evaluation Low AUC

## Date

2026-05-10 14:48

## Run ID

`2026-05-10_1448_eval-fast-low-auc`

## Experiment

Evaluate the original baseline checkpoint through the patched `New_baseline` evaluator using the fast default sequence cap.

## Configuration

- `EVAL_NUM_WORKERS=4`
- `EVAL_MERGE_BATCHES=0`
- Effective sequence lengths: `seq_a:256,seq_b:256,seq_c:256,seq_d:256`
- AMP: BF16

## Result

- Platform AUC: `0.811994`
- Platform inference time: `61.5s`
- Internal inference time: `61.081s`
- Predictions: `310000`
- Unique users: `310000`

## Diagnosis

The run is far below the inference-time budget and has the lowest AUC so far. The likely quality loss is from truncating `seq_c` and `seq_d` from `512` to `256`. Throughput is now healthy, so the next useful direction is to spend more time on longer sequence windows until close to the inference-time limit.

## Fix

Change `New_baseline/infer.py` default effective sequence lengths to:

`seq_a:256,seq_b:256,seq_c:384,seq_d:384`

This is a middle point between the fast low-AUC run and the full-window run that completed internally in `1890.286s`.

## Sweep Plan

Evaluate in this order, stopping once runtime is close to the platform limit:

1. `seq_a:256,seq_b:256,seq_c:384,seq_d:384`
2. `seq_a:256,seq_b:256,seq_c:448,seq_d:448`
3. `seq_a:256,seq_b:256,seq_c:512,seq_d:512`

Keep `EVAL_NUM_WORKERS=4`, `EVAL_MERGE_BATCHES=0`, and BF16 unless logs show a new bottleneck.
