# Codex Workflow

This is the only prompt file that needs to be read first.

## 1. Mission

Your job is to help operate this repository for TAAC2026 experiments.

Important current status:

- The `baseline/` split is already confirmed to run successfully.
- That baseline has produced one trained model and one successful evaluation result with score `0.8133`.
- The active goal is not baseline setup. The active goal is to modify the model training code and evaluation method to beat `0.8133`.
- Treat `0.8133` as the current benchmark until a newer feedback record says otherwise.

The workflow is:

1. Train a model from the user-provided training code.
2. Evaluate that model with the user-provided evaluation code.
3. Record the result, logs, and next action.

## 2. Competition Rules That Matter

- Training has no fixed attempt limit, but each run is expensive and can take hours.
- At most two models may train at the same time.
- Evaluation allows only three successful submissions per day.
- Every successful evaluation returns a score result plus terminal logs.
- Never spend an evaluation slot on an unverified checkpoint.

## 3. Directory Roles

### Root-level core files

- `main.py`: primary training entry point.
- `infer.py`: primary inference entry point.
- `dataset.py`: shared dataset logic.
- `model.py`: shared model definition.
- `run.sh`: main training launcher.
- `run_taac2026.sh`: TAAC2026 launcher.

### Baseline split

- `baseline/training/`: training-only baseline copy.
- `baseline/evaluation/`: evaluation-only baseline copy.
- `baseline/README.md`: pointer for the split.
- Current known baseline status: one trained model and one evaluation result, score `0.8133`.
- Active code changes should happen under `baseline/training/` and `baseline/evaluation/`.
- Treat root-level Python files as compatibility wrappers or legacy code unless the user explicitly asks to change them.

### Planning and process

- `prompt/`: the working process hub.
- `submission/runs/`: immutable code snapshots for each run id.
- `feedbacks/runs/`: run-linked feedback, logs, and error notes.
- `feedbacks/records/`: chronological run history with metrics and links.
- `docs/planning/`: longer planning notes and archival prompts.

### References

- `docs/`: competition notes, templates, and writeups.
- `champion_plan/`: reference solutions and external ideas.
- `original/`: original source snapshot.

## 4. How To Work

- Keep training code and evaluation code separate.
- Make all new model and evaluation operations under the baseline split folders.
- Use `baseline/training/` for model/training changes.
- Use `baseline/evaluation/` for inference/evaluation-method changes.
- Keep documentation in English.
- Make the directory layout easy to scan.
- Record every meaningful run in chronological order.
- Update the prompt files when the process changes.

## 5. Training Flow

Use this order for training work:

1. Inspect the active training code and current configuration.
2. Run a short smoke experiment first.
3. Change only one important variable at a time.
4. Compare validation accuracy, runtime, and stability.
5. Keep only the strongest checkpoint candidates.

## 6. Evaluation Flow

Every competition day, the first job is to define and execute the three available evaluation tasks. Do this before training submissions, architecture implementation, or longer research work.

Daily evaluation discipline:

1. Read `prompt/plan.md` and identify the three evaluation packages/tasks for the day.
2. Submit evaluation task 1.
3. Record AUC, inference time, package path, logs, and promotion decision.
4. Update `prompt/plan.md` immediately.
5. Choose task 2 based on the updated result.
6. Repeat until all three daily evaluation slots are used or explicitly deferred by the user.

Before any platform submission:

1. Run local inference.
2. Confirm the output file exists.
3. Confirm the JSON structure is correct.
4. Confirm user IDs and prediction counts are aligned.
5. Confirm values are numeric and non-negative.
6. Run the local verifier if available.

Only after that should an evaluation slot be used.

## 7. Promotion Criteria

A checkpoint may be promoted only if:

- local inference succeeds,
- the output format matches the platform contract,
- the validation result is clearly competitive,
- the gain is stable rather than a one-off spike.

## 8. Feedback Logging

Use one shared run id for code, logs, and feedback, with the filename pattern `YYYY-MM-DD_HHMM_short-title`.

For each meaningful run:

- copy the submitted training code snapshot into `submission/runs/<run_id>/train/`,
- copy the submitted evaluation code snapshot into `submission/runs/<run_id>/eval/`,
- write the detailed train/eval/error notes under `feedbacks/runs/<run_id>/`,
- write the short chronological summary into `feedbacks/records/<run_id>.md`.

Each summary record should include:

- date and time,
- experiment name,
- configuration,
- accuracy or score,
- key log notes,
- decision,
- next action,
- links to the matching code and feedback folders.

## 8.1 Current Day Handoff

Completed on 2026-05-08:

- Baseline submission path was exercised.
- Evaluation submission code was made self-contained and instrumented with `print` feedback logs.
- Timeout and NaN failures were diagnosed from platform logs and patched in `baseline/evaluation/infer.py`.
- A separate `LOSS_TYPE=bce_pairwise` training path was added for tomorrow's improvement experiment.

Tomorrow's first actions:

1. Resubmit the patched baseline evaluation path if the latest platform run has not yet produced a clean score.
2. Inspect the platform logs for `PLATFORM_INFER_SUMMARY` and any `PLATFORM_SCORE_SANITIZE` entries.
3. Compare the returned score with `0.8133`.
4. Move to the `bce_pairwise` experiment only after the baseline path is stable.

## 9. Current Experiment Discipline

- Do not spend time re-proving that the baseline runs; it already has a successful model and evaluation.
- Every model or evaluation change should be judged against the `0.8133` baseline score.
- Keep the baseline architecture stable unless a small change clearly improves validation.
- Treat sequence truncation as the first experiment axis.
- Favor short runs before long ones.
- Keep only the next action in focus.
- Update the plan as soon as feedback changes the next step.

## 10. Where To Look Next

If you need more detail, read these files in order:

1. `prompt/plan.md`
2. `prompt/strategy.md`
3. `feedbacks/records/template.md`

Those files are supporting notes. `prompt/codex_workflow.md` is the master entry point.
