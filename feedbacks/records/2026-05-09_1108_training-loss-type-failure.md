# Training Loss Type Failure

## Date

2026-05-09 11:08

## Experiment

Today's new training run using the `bce_pairwise` candidate.

## Configuration

- Submitted training archive: `submission/platform_uploads/2026-05-09_training_bce_pairwise.zip`
- Launcher requested: `LOSS_TYPE=bce_pairwise`
- Pairwise config: `PAIRWISE_AUC_WEIGHT=0.05`, `PAIRWISE_MAX_PAIRS=8192`
- Failure log: `feedbacks/logs_output/error4.md`

## Result

Training failed before model construction.

## Log Notes

- The platform copied `run.sh`, `dataset.py`, `model.py`, `ns_groups.json`, `train.py`, `trainer.py`, and `utils.py`.
- The launcher printed `loss_type":"bce_pairwise"`.
- The executed `train.py` failed argument parsing: `invalid choice: 'bce_pairwise' (choose from 'bce', 'focal')`.
- This indicates the runtime parser did not expose the pairwise loss path even though the local archive contains the updated parser.
- Structured failure notes were written to `feedbacks/logs_output/error/2026-05-09_1108_training-loss-type-failure.md` and `feedbacks/platform_runs/2026-05-09_1108_training-loss-type-failure.md`.

## Decision

Prepare a retry that cannot fail on this parser mismatch. The updated launcher checks whether runtime `train.py` contains `bce_pairwise`; if not, it falls back to `LOSS_TYPE=focal` and avoids passing pairwise-only arguments.

## Next Action

Submit `submission/platform_uploads/2026-05-09_training_pairwise-or-focal-retry.zip` as the replacement training run. If it logs `PLATFORM_TRAIN_LOSS_FALLBACK`, record the run as the focal-loss candidate rather than the pairwise candidate.
