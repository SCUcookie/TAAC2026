# Archive Fast Eval And 2-GPU Training Recipe

## Date

2026-05-11 15:33

## Result Context

Two evaluations of the same 2-GPU checkpoint were compared from `temp.md`.

- Changed `New_baseline` evaluator: AUC `0.818008`, inference about `1000s`
- Archived/original evaluator: AUC `0.818178`, inference about `400s`

## Interpretation

The best score and runtime come from pairing the 2-GPU checkpoint with the archived single-pass evaluator.

The changed evaluator was slower and slightly worse because it added recent-blend TTA, capped `seq_d` to `448`, and performed extra output/instrumentation work. The archived evaluator keeps the checkpoint's full train-time sequence lengths and runs one clean forward pass.

## Action

Implemented the plan by aligning `New_baseline` to the archived high-confidence path:

- Restored `New_baseline/infer.py` to the archived single-pass evaluator.
- Restored `New_baseline/model.py` to the archived model definition.
- Restored `New_baseline/run.sh` to the archived training launcher with `hash`, EMA, full sequence defaults, and `num_workers=8`.
- Created immutable snapshots under `submission/runs/2026-05-11_1533_archive-fast-eval/`.
- Prepared platform packages for the remaining evaluation chance and the next training job.

## Next Action

Use the eval package for the final 2026-05-11 evaluation chance against the best 2-GPU checkpoint, then submit the training package for the next 2-GPU training job.
