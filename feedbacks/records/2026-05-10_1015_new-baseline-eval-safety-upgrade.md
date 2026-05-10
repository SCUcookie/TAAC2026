# New Baseline Evaluation Safety Upgrade

## Date

2026-05-10 10:15

## Run ID

`2026-05-10_1015_new-baseline-eval-safety-upgrade`

## Experiment

Improve `New_baseline/infer.py` while the OOM-safe training retry is running, without enabling the previous memory-heavy recent-window TTA path.

## Changes

- Added independent inference controls: `EVAL_BATCH_SIZE`, `EVAL_NUM_WORKERS`, `EVAL_USE_AMP`, `EVAL_AMP_DTYPE`, and `EVAL_PROGRESS_EVERY`.
- Added rank-changing sequence-window controls: `EVAL_SEQ_MAX_LENS` and `EVAL_SEQ_LEN_CAP`.
- Added platform events for checkpoint status, inference config, dataset config, AMP config, first batch, progress, sanitization, and final summary.
- Added BF16/FP16 autocast with fp32 fallback if non-finite scores appear.
- Added score sanitization, duplicate user-id checks, prediction-count checks, and finite non-negative score checks.
- Added auxiliary outputs: `submission.csv`, `scores.npy`, and `inference_summary.json`.
- Kept checkpoint construction driven by `train_config.json`, including x-domain/time features and new model flags.

## Evaluation Method

Use the default evaluator first for correctness. If the new checkpoint passes the local training gate, test a lightweight rank-changing variant by setting `EVAL_SEQ_MAX_LENS` or `EVAL_SEQ_LEN_CAP`. This changes the input window and ranking with only one forward pass, avoiding the CUDA OOM risk from two-pass TTA.

## Submission Gate

Do not spend a platform slot until the evaluator emits `PLATFORM_INFER_SUMMARY`, returns exactly `310000` unique predictions, and the checkpoint has passed the local AUC gate.
