# Evaluation Repeat Calibration

## Date

2026-05-10 16:15

## Run ID

`2026-05-10_1615_eval-273s-repeat-calibration`

## Observation

The `recent_blend` evaluation with `EVAL_TTA_REPEAT=5` finished with platform
inference time `273.55s`.

The target runtime is about `1300s`, close to the evaluation time limit.

## Runtime Estimate

The previous no-TTA near-full-window run took `186.62s`.

Estimated extra time per repeated recent pass:

`(273.55 - 186.62) / 5 = 17.386s`

A repeat count of `60` gives an estimated runtime:

`186.62 + 60 * 17.386 = 1229.78s`

This is close to the `1300s` target while leaving some margin below the limit.

## Action

Set the packaged default:

`EVAL_TTA_REPEAT=60`

Keep:

- `EVAL_TTA_MODE=recent_blend`
- `EVAL_TTA_WEIGHT=0.12`
- `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- `EVAL_SEQ_MAX_LENS=seq_a:256,seq_b:256,seq_c:512,seq_d:448`

## Next Sweep

If the platform runtime is still below `1150s`, try `EVAL_TTA_REPEAT=65`.

If the platform runtime is above `1280s`, try `EVAL_TTA_REPEAT=55`.
