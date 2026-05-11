# Tomorrow Startup Prompt

Read `prompt/codex_workflow.md`, then `prompt/plan.md`, then the latest records under `feedbacks/records/`.

Work in `New_baseline/` unless the user redirects. Preserve the baseline checkpoint and model weights. Only modify evaluation-side code.

Current state:

- Best result remains the original baseline checkpoint with AUC `0.813094` and inference time `1480.38s`.
- Best known config is `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:448`, `EVAL_TTA_MODE=recent_blend`, `EVAL_TTA_WEIGHT=0.12`, `EVAL_TTA_REPEAT=14`, and `EVAL_TTA_CROP_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`.
- A newly trained model scored only AUC `0.80424` with inference time `907.78s` under the same best known evaluator, so it should not replace the original baseline checkpoint.
- The user may retrain that model with 2 GPUs because the one-GPU attempt hit OOM. If redirected to training, treat it as a separate experiment and still compare against the original baseline checkpoint.

First inspect current `New_baseline/infer.py` and `New_baseline/dataset.py`.

Implement true tail-window inference as a controlled evaluation variant:

- Current dataset prefix truncation uses `values[start:start + use_len]`.
- Add an eval-only way to read suffix/tail windows using `values[end - use_len:end]`.
- Keep prefix/full-window behavior available.
- Blend logits from full/prefix and true-tail views.

Initial candidate:

- Full-window logit weight `0.85`
- Tail-window logit weight `0.15`
- Keep BF16 AMP unless logs show non-finite scores.
- Target runtime can approach the limit if AUC improves.

Do not repeat identical TTA passes for AUC. Repeating deterministic logits does not change rank order meaningfully.

Useful variants to test, in order:

1. Full-window + true-tail blend, weight `0.15`
2. Weight sweep: `0.08`, `0.12`, `0.20`
3. Multi-window blend: prefix + middle + tail, only if tail blend helps
4. FP32/no-AMP comparison only if there is enough evaluation budget

Before platform submission:

1. Run syntax check on changed Python files.
2. Confirm inference config logs include the active window mode and blend weights.
3. Confirm output has exactly `310000` predictions and `310000` unique user IDs.
4. Confirm scores are finite and non-negative.
5. Confirm platform log reaches `PLATFORM_INFER_SUMMARY`.

Record every submitted result in `feedbacks/records/`.
