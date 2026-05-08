# Platform Dataset Probe

Upload this directory to the evaluation platform when you want to inspect the
evaluation dataset layout instead of running a real model.

Files produced under `EVAL_RESULT_PATH`:

- `predictions.json`: valid fallback predictions for all discovered `user_id`s
- `platform_env.json`: relevant platform environment variables
- `dataset_summary.json`: dataset structure and sampled statistics
- `dataset_summary.txt`: readable summary

This probe is intentionally self-contained and does not depend on a trained
checkpoint.
