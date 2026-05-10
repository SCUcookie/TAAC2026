# OOM-Safe Pairwise Training Retry

## Date

2026-05-09 15:25

## Run ID

`2026-05-09_1525_training-oom-safe-pairwise`

## Code Snapshot

`submission/runs/2026-05-09_1525_training-oom-safe-pairwise/train/`

## Feedback Folder

`feedbacks/platform_runs/2026-05-09_1525_training-oom-safe-pairwise.md`

## Experiment

Retry the failed model-training process only. The goal is to train a new model candidate with the RankMixer and pairwise-loss improvement path while avoiding the CUDA OOM seen in the previous platform run.

## Configuration

- Upload archive: `submission/platform_uploads/2026-05-09_training_oom-safe-pairwise.zip`
- `BATCH_SIZE=64`
- `SEQ_ENCODER_TYPE=longer`
- `SEQ_TOP_K=50`
- `SEQ_MAX_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- `NUM_WORKERS=4`
- `BUFFER_BATCHES=8`
- `LOSS_TYPE=bce_pairwise`, with focal fallback if runtime parser support is missing

## Result

Archive prepared; platform submission pending.

## Log Notes

The previous structured failure log shows first-forward CUDA OOM with `BATCH_SIZE=256`, `SEQ_ENCODER_TYPE=transformer`, and long windows up to 512. This retry reduces activation memory and keeps the pairwise objective.

## Decision

Use this archive for the next training submission instead of reusing `2026-05-09_training_pairwise-or-focal-retry.zip`.

## Next Action

Submit `submission/platform_uploads/2026-05-09_training_oom-safe-pairwise.zip` as the next training job. Do not start evaluation until the platform produces a checkpoint.

## Links

- Submission train snapshot: `submission/runs/2026-05-09_1525_training-oom-safe-pairwise/train/`
- Submission eval snapshot: not involved
- Full run feedback: `feedbacks/platform_runs/2026-05-09_1525_training-oom-safe-pairwise.md`
