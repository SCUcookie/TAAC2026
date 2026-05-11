# Evaluation-Only AUC Plan

## Current Best

- Current best evaluation result: AUC `0.813094`, inference time `1480.38s`.
- Best checkpoint remains the original baseline checkpoint under the best known evaluator.
- AUC is the priority; high latency close to the platform limit is acceptable if it improves AUC.
- Preserve the original baseline checkpoint and model weights.
- For evaluation improvement work, stop changing training code. Only evaluation-side code should change unless the user explicitly redirects to a new training run.

## Latest New-Model Result

- The newly trained model was evaluated with the best known evaluation method that produced AUC `0.813094` on the original baseline checkpoint.
- Result: AUC `0.80424`, inference time `907.78s`.
- Interpretation: this checkpoint is not competitive with the original baseline checkpoint and should not replace it for evaluation-only tuning.
- User plans to retrain this model with 2 GPUs because the one-GPU attempt hit OOM.
- Keep the original baseline checkpoint as the score target and fallback while any 2-GPU retraining experiment runs.

## Best Known Config

- `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:448`
- `EVAL_TTA_MODE=recent_blend`
- `EVAL_TTA_WEIGHT=0.12`
- `EVAL_TTA_REPEAT=14`
- `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`

## Active Direction

The next evaluation-only AUC direction is to implement true raw-sequence tail-window scoring and blend it with the existing full-window score.

Do not spend more effort tuning `EVAL_TTA_REPEAT`; it mainly changes latency, not ranking. Repeating deterministic logits does not meaningfully change rank order.

## Next Implementation

Work in `New_baseline/` unless the user redirects.

1. Inspect current `New_baseline/infer.py` and `New_baseline/dataset.py`.
2. Add an eval-only way to read suffix/tail windows from raw sequences using `values[end - use_len:end]`.
3. Keep the existing prefix/full-window behavior available. The current dataset prefix truncation uses `values[start:start + use_len]`.
4. Blend logits from the full/prefix view and the true-tail view.
5. Log the active window mode and blend weights in inference output.
6. Record every submitted result in `feedbacks/records/`.

## Initial Candidate

- Full-window logit weight: `0.85`
- Tail-window logit weight: `0.15`
- Keep BF16 AMP unless logs show non-finite scores.
- Target runtime can approach the limit if AUC improves.

## Useful Variants To Test

1. Full-window + true-tail blend, tail weight `0.15`.
2. Weight sweep: `0.08`, `0.12`, `0.20`.
3. Multi-window blend: prefix + middle + tail, only if tail blend helps.
4. FP32/no-AMP comparison only if there is enough evaluation budget.

## Pre-Submission Test Plan

1. Run syntax check on changed Python files.
2. Confirm inference config logs include the active window mode and blend weights.
3. Confirm output has exactly `310000` predictions and `310000` unique user IDs.
4. Confirm scores are finite and non-negative.
5. Confirm platform log reaches `PLATFORM_INFER_SUMMARY`.
