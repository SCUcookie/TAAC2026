# New Baseline Training Launch Plan

## Date

2026-05-10 10:00

## Run ID

`2026-05-10_1000_new-baseline-training-launch-plan`

## Experiment

Train the new model package from `New_baseline/` after rejecting the 2026-05-09 checkpoint with local AUC `0.859`.

## Configuration

- Entry script: `New_baseline/run.sh`
- Active recipe: RankMixer tokenizer, wide dense tokens, signlog dense transform, x-domain features, inter-time buckets, continuous time, target-aware attention, cross-domain attention, SwiGLU FFN, RMSNorm, EMA, random ID masking
- Platform paths required: `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`
- Local data status: no local TAAC2026 `.parquet` files or `schema.json` were found under this workspace, so local launch would fail before training.

## Command

```bash
cd New_baseline
bash run.sh
```

Use the platform-provided environment variables for the dataset, checkpoint, logs, and TensorBoard event directory.

## Gate

Do not submit the resulting checkpoint unless validation AUC beats the original baseline local gate of `0.861`, or unless a separate reason to override the gate is recorded before submission.

## Decision

Training should be launched on the platform or another environment that has the TAAC2026 parquet training data. This local workspace is suitable for code review and packaging but does not contain the training dataset.

## Next Action

After the platform training finishes, record the best checkpoint path, best validation AUC, logloss, runtime, and whether a `.best_model/model.pt` checkpoint was produced.
