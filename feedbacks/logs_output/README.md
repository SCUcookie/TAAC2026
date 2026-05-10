# Raw Logs

This folder is for raw terminal output only.

New layout:

- `feedbacks/logs_output/train/`: training logs.
- `feedbacks/logs_output/eval/`: evaluation logs.
- `feedbacks/logs_output/error/`: stack traces and hard failures.

Rules:

- Do not keep mixed train/eval/error output in one flat file.
- Name each log with the same run id used in `submission/runs/` and `feedbacks/runs/`.
- Keep legacy flat files only as archive material; new runs should use the split folders.
