# Launch Training And Eval1

## Date

2026-05-09 10:48

## Experiment

Start today's new training run and first platform evaluation submission concurrently.

## Configuration

- Training upload archive: `submission/platform_uploads/2026-05-09_training_bce_pairwise.zip`
- Training code path: `baseline/training/`
- Training launcher: `run.sh`
- Training loss: `LOSS_TYPE=bce_pairwise`
- Pairwise config: `PAIRWISE_AUC_WEIGHT=0.05`, `PAIRWISE_MAX_PAIRS=8192`
- Other training defaults: rankmixer tokenizer, `USER_NS_TOKENS=5`, `ITEM_NS_TOKENS=2`, `NUM_QUERIES=2`, `D_MODEL=64`, `NUM_HEADS=4`, `NUM_HYFORMER_BLOCKS=2`, `SEQ_MAX_LENS=seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- Eval1 upload archive: `submission/platform_uploads/2026-05-09_eval1_original-baseline_bs2048_fp32.zip`
- Eval1 model: original baseline checkpoint, stronger known checkpoint under original evaluator
- Eval1 evaluator: patched `baseline/evaluation/infer.py`
- Eval1 inference defaults: full precision, `EVAL_BATCH_SIZE=2048`, `EVAL_USE_AMP=0`

## Result

Training submission failed quickly, and Eval1 also timed out on the original baseline checkpoint.

## Log Notes

- `baseline/evaluation/infer.py` default batch size was raised from `1024` to `2048` to avoid repeating the timeout seen in `feedbacks/logs_output/log4.md`.
- `baseline/training/run.sh` was added so the training lane can be uploaded independently and starts the `bce_pairwise` candidate by default.
- Python parse check passed for the changed entry points with `python -B` and `ast.parse`.
- Standard `py_compile` was not usable locally because Windows denied writing a `__pycache__` replacement file.
- Training failure log copied to `feedbacks/logs_output/error4.md`.
- Failure cause: runtime `train.py` rejected `--loss_type=bce_pairwise`, reporting choices `bce` and `focal`.
- A retry archive was prepared with a robust launcher that falls back to `focal` if the runtime parser lacks `bce_pairwise` support: `submission/platform_uploads/2026-05-09_training_pairwise-or-focal-retry.zip`.
- Eval1 used the original baseline checkpoint and hit the platform timeout before `PLATFORM_INFER_SUMMARY` was emitted.
- Structured eval logs were written to `feedbacks/logs_output/eval/2026-05-09_1048_eval1-original-baseline-timeout.md` and `feedbacks/logs_output/error/2026-05-09_1048_eval1-original-baseline-timeout.md`.

## Decision

Submit the two prepared archives concurrently through the platform:

1. Retry the training job from `2026-05-09_training_pairwise-or-focal-retry.zip`.
2. Start Eval1 from `2026-05-09_eval1_original-baseline_bs2048_fp32.zip`, selecting the original baseline checkpoint.

## Next Action

After Eval1 returns, inspect the platform log for `PLATFORM_INFER_SUMMARY`, score sanitization counts, and the returned score. If Eval1 finishes cleanly, submit Eval2 for the newly trained existing model.
