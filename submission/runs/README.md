# Submission Run Snapshots

Store one immutable code snapshot per run id here.

Recommended structure:

- `submission/runs/<run_id>/train/`: training-side code that was submitted for that run.
- `submission/runs/<run_id>/eval/`: evaluation-side code that was submitted for that run.
- `submission/runs/<run_id>/shared/`: optional shared files copied into both sides.
- `submission/runs/<run_id>/manifest.md`: exact entry points, commands, and checkpoint paths.

Rules:

- Never overwrite an existing run folder.
- Never reuse a run id for a different code snapshot.
- Keep train and eval code inside the same run folder so they remain linked.
- Copy the exact files that were sent to the platform, not the working tree by reference.
