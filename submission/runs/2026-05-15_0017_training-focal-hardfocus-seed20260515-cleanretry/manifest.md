# Training Focal Hard-Focus Seed 20260515 Clean Retry

## Date

2026-05-15

## Purpose

Clean retry of the focal hard-focus training package after the previous platform run failed before training due to a file mapping/package issue.

## Training Package

Directory:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry/`

Zip:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry.zip`

## Controlled Changes

Same intended training change as `0016`:

- base: promoted `20260512` architecture
- seed: `20260515`
- loss: `focal`
- `focal_alpha`: `0.25`
- `focal_gamma`: `1.5`
- no recent-crop augmentation
- evaluator compatibility unchanged

## Packaging Checks

- zip contains exactly: `dataset.py`, `model.py`, `ns_groups.json`, `run.sh`, `train.py`, `trainer.py`, `utils.py`
- `dataset.py` starts with the Python module docstring
- `run.sh` starts with `#!/bin/bash`
- Python source syntax check passed

## After Training

Evaluate only if local validation is competitive with `0.866320891869`.
