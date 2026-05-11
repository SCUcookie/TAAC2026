# 2026-05-10 End-of-Day Summary

## Goal

Improve platform AUC while preserving the strongest known baseline checkpoint. Today's work focused on evaluation-side tuning, then checked whether a newly trained model could replace the original baseline checkpoint.

## Best Result

- Best platform AUC remains `0.813094`.
- Inference time for that run was `1480.38s`.
- Best checkpoint remains the original baseline checkpoint.
- Best known evaluation config:
  - `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:448`
  - `EVAL_TTA_MODE=recent_blend`
  - `EVAL_TTA_WEIGHT=0.12`
  - `EVAL_TTA_REPEAT=14`
  - `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`

## Evaluation Work

- Multiple faster evaluation variants were tested during the day, but the faster variants gave lower AUC.
- Increasing deterministic repeat count mainly changed runtime, not ranking quality.
- `EVAL_TTA_REPEAT=60` was too slow for the platform budget.
- `EVAL_TTA_REPEAT=14` was the best known balance today, reaching AUC `0.813094` while still close to the platform time limit.
- Conclusion: do not spend more effort tuning deterministic repeat count. The next useful evaluation-only lever should create a genuinely different view of the raw sequence.

## Next Evaluation Direction

Implement true raw-sequence tail-window inference and blend it with the current full/prefix-window score.

- Current prefix truncation behavior uses `values[start:start + use_len]`.
- Add an eval-only tail mode using `values[end - use_len:end]`.
- Keep the existing full/prefix behavior available.
- Blend logits from full/prefix and true-tail views.
- Initial candidate: full-window logit weight `0.85`, tail-window logit weight `0.15`.

## New Training Result

- A newly trained model finished training and was evaluated with the same best known evaluation method.
- Result: AUC `0.80424`, inference time `907.78s`.
- This is far below the original baseline checkpoint's `0.813094`, so it should not replace the original baseline checkpoint.
- The user plans to train this model again with 2 GPUs because the one-GPU attempt hit OOM.

## Tomorrow Priorities

1. Keep the original baseline checkpoint as the score target and fallback.
2. If continuing evaluation-only work, implement true tail-window scoring in `New_baseline/infer.py` and `New_baseline/dataset.py`.
3. If redirected to training, treat the 2-GPU run as a separate experiment and compare the resulting checkpoint against the original baseline using the same best known evaluator.
4. Record every submitted result in `feedbacks/records/`.

## Verification Checklist For Submissions

- Syntax check changed Python files.
- Confirm inference logs include active window mode and blend weights.
- Confirm output has exactly `310000` predictions and `310000` unique user IDs.
- Confirm scores are finite and non-negative.
- Confirm platform log reaches `PLATFORM_INFER_SUMMARY`.
