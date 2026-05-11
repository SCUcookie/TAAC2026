# Evaluation Repeat 14 Best AUC But Slow

## Date

2026-05-10 17:30

## Run ID

`2026-05-10_1730_eval-repeat14-best-auc-slow`

## Result

The evaluation process finished successfully.

- AUC: `0.813094`
- Platform inference time: `1480.38s`
- `EVAL_TTA_MODE=recent_blend`
- `EVAL_TTA_WEIGHT=0.12`
- `EVAL_TTA_REPEAT=14`
- `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:448`

## Comparison

This is slightly higher than the previous best known submitted AUC:

- Previous: `0.813041`
- Current: `0.813094`
- Gain: `+0.000053`

The inference time is high, though, and should be moved closer to the `1300s`
target.

## Action

Reduce packaged default:

`EVAL_TTA_REPEAT=12`

Runtime estimate:

`1480.38 * 13 / 15 = 1283.0s`

This keeps the same `recent_blend` scoring method but lowers repeated recent
passes from `14` to `12`.

## Next Sweep

Submit with `EVAL_TTA_REPEAT=12`.

If AUC drops materially, keep `repeat=14` as the stronger but slower setting.
If runtime lands too low, try `repeat=13`.
