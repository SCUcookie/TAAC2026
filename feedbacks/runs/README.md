# Run-Linked Feedback

Store the detailed notes for each run here, using the same run id as the matching snapshot under `submission/runs/`.

Recommended structure:

- `feedbacks/runs/<run_id>/train.md`: training notes, metrics, and failure modes.
- `feedbacks/runs/<run_id>/eval.md`: evaluation notes, score, and platform result.
- `feedbacks/runs/<run_id>/error.md`: hard errors, stack traces, and platform failures.
- `feedbacks/runs/<run_id>/logs/`: split raw logs if you want to keep them alongside the notes.

Rules:

- Use one run id for code, logs, and feedback.
- Keep training and evaluation separate, but never detached from the same run folder.
- Put mixed or ambiguous output into the appropriate train/eval/error file instead of a single flat log.
- Keep the chronological summary in `feedbacks/records/` and the full notes here.
