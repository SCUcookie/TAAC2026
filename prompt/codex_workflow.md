# Codex Workflow

This is the only prompt file that needs to be read first.

## 1. Mission

Your job is to help operate this repository for TAAC2026 experiments.

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

### Planning and process

- `prompt/`: the working process hub.
- `feedbacks/records/`: chronological run history with metrics and logs.
- `docs/planning/`: longer planning notes and archival prompts.

### References

- `docs/`: competition notes, templates, and writeups.
- `champion_plan/`: reference solutions and external ideas.
- `original/`: original source snapshot.

## 4. How To Work

- Keep training code and evaluation code separate.
- Prefer the baseline split folders when reasoning about those two workflows.
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

Write one record per meaningful run into `feedbacks/records/`.

Each record should include:

- date and time,
- experiment name,
- configuration,
- accuracy or score,
- key log notes,
- decision,
- next action.

Use the filename pattern `YYYY-MM-DD_HHMM_short-title.md`.

## 9. Current Experiment Discipline

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