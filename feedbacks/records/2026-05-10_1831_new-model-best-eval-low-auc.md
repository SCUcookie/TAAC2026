# New Model Evaluated With Best Known Evaluator

## Result

- Date: `2026-05-10`
- Model: newly trained model
- Evaluation method: best known evaluation method from the original baseline checkpoint run
- Prior best with original baseline checkpoint: AUC `0.813094`, inference time `1480.38s`
- New model result: AUC `0.80424`, inference time `907.78s`

## Interpretation

The new checkpoint is much weaker than the original baseline checkpoint under the same best known evaluation setup. It should not replace the original baseline checkpoint for evaluation-only AUC tuning.

## Next Action

The user plans to retrain this model with 2 GPUs because the one-GPU training attempt hit OOM. Keep the original baseline checkpoint as the score target and fallback during this retraining path.
