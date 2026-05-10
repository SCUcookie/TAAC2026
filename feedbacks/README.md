# Feedbacks

This folder stores run-linked feedback and logs.

The master workflow prompt is in `prompt/codex_workflow.md`.

## Canonical layout

- `feedbacks/records/`: short chronological summaries.
- `feedbacks/runs/<run_id>/`: full run notes, split into train, eval, and error files.
- `feedbacks/logs_output/`: raw terminal output, now separated by log type.

## Rules

- Use one run id for the code snapshot, the logs, and the feedback.
- Keep training and evaluation notes in the same run folder, but separate files.
- Put hard failures in `error/` instead of mixing them into the main log.
- Keep the files in date order.
- Use English only.
