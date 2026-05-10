# Current Workflow File Map

This document explains the current experiment workflow at file level.

## 1. Source Of Truth

- `prompt/codex_workflow.md`: master operating prompt.
- `baseline/training/`: active training code path.
- `baseline/evaluation/`: active evaluation code path.
- `submission/platform_uploads/`: zipped archives uploaded to the platform.
- `submission/runs/`: immutable snapshots of what was submitted.
- `feedbacks/platform_runs/`: detailed platform run reports.
- `feedbacks/logs_output/train/`: raw training terminal logs.
- `feedbacks/logs_output/eval/`: raw evaluation terminal logs.
- `feedbacks/logs_output/error/`: raw failure logs and stack traces.
- `feedbacks/records/`: short chronological summaries.
- `feedbacks/runs/`: optional per-run folders for longer notes.

## 2. Current Runtime Flow

### Training path

1. Edit model and training logic in `baseline/training/`.
2. Create a run id using `YYYY-MM-DD_HHMM_short-title`.
3. Copy the exact training snapshot into `submission/runs/<run_id>/train/`.
4. Build the upload archive in `submission/platform_uploads/<archive_name>.zip`.
5. Submit the archive to the platform.
6. Save the platform report in `feedbacks/platform_runs/<run_id>.md`.
7. Save the raw terminal output in `feedbacks/logs_output/train/<run_id>.md` when the run reaches training execution, or `feedbacks/logs_output/error/<run_id>.md` if it fails early.
8. Write the summary entry in `feedbacks/records/<run_id>.md`.

### Evaluation path

1. Edit inference and evaluation logic in `baseline/evaluation/`.
2. Reuse the checkpoint produced by the matching training run.
3. Copy the exact evaluation snapshot into `submission/runs/<run_id>/eval/`.
4. Build the evaluation upload archive in `submission/platform_uploads/<archive_name>.zip`.
5. Submit the archive to the platform.
6. Save the platform report in `feedbacks/platform_runs/<run_id>.md`.
7. Save the raw evaluation output in `feedbacks/logs_output/eval/<run_id>.md`.
8. If the run fails, move the failure trace into `feedbacks/logs_output/error/<run_id>.md`.
9. Write the summary entry in `feedbacks/records/<run_id>.md`.

## 3. One-To-One Rules

- One run id must map to one submitted code snapshot.
- A training run and its paired evaluation run belong to the same experiment family, but they are still written as separate train/eval artifacts under the same run id.
- The evaluation model must always point to a unique training checkpoint.
- Never overwrite a run folder after the archive has been generated.
- Keep train, eval, and error logs separate.
- Keep old flat logs only as archive material.

## 4. Today\'s Files

### Training failure

- Submitted archive: `submission/platform_uploads/2026-05-09_training_bce_pairwise.zip`
- Retry archive: `submission/platform_uploads/2026-05-09_training_pairwise-or-focal-retry.zip`
- Raw failure log: `feedbacks/logs_output/error4.md`
- New structured failure log: `feedbacks/logs_output/error/2026-05-09_1108_training-loss-type-failure.md`
- Platform report: `feedbacks/platform_runs/2026-05-09_1108_training-loss-type-failure.md`
- Summary record: `feedbacks/records/2026-05-09_1108_training-loss-type-failure.md`

### Evaluation failure

- Submitted archive: `submission/platform_uploads/2026-05-09_eval1_original-baseline_bs2048_fp32.zip`
- Model used: original baseline checkpoint
- Raw timeout log: `feedbacks/logs_output/log4.md`
- New structured evaluation log: `feedbacks/logs_output/eval/2026-05-09_1048_eval1-original-baseline-timeout.md`
- New structured error log: `feedbacks/logs_output/error/2026-05-09_1048_eval1-original-baseline-timeout.md`
- Platform report: `feedbacks/platform_runs/2026-05-09_1048_eval1-original-baseline-timeout.md`
- Summary record: `feedbacks/records/2026-05-09_1048_launch-training-and-eval1.md`

## 5. What To Update Next

- If a new run is launched, allocate a new run id immediately.
- If training finishes, record the checkpoint path before starting any evaluation.
- If evaluation fails, copy the exact failure text into `feedbacks/logs_output/error/` and update the matching `feedbacks/platform_runs/` file.
- If the model or archive changes, create a new run id instead of rewriting the old one.
