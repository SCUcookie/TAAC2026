# New Baseline Evaluation State-Dict Mismatch

## Date

2026-05-10 10:56

## Run ID

`2026-05-10_1056_new-baseline-eval-state-dict-mismatch`

## Experiment

Evaluation submission used uploaded `infer.py`, `train.py`, `model.py`, and `dataset.py` from `New_baseline/`, but selected the older original baseline checkpoint.

## Result

Evaluation failed before first batch inference during strict checkpoint loading.

Missing keys included:

- `user_dense_tokenizer.scalar_proj.*`
- `blocks.*.mixer.ffn.*`

Unexpected keys included:

- `user_dense_proj.*`
- `blocks.*.mixer.fc1.*`
- `blocks.*.mixer.fc2.*`

## Diagnosis

The selected checkpoint was trained with the older model class. The uploaded `New_baseline/model.py` expected the newer dense tokenizer and generic mixer FFN module names. The checkpoint itself was present and readable; dataset construction also succeeded.

## Fix

Add legacy compatibility flags in `New_baseline/model.py` and infer them in `New_baseline/infer.py` when old `train_config.json` files lack the newer dense-tokenizer and FFN config keys.

## Next Action

Resubmit the patched evaluation package if evaluating the old original checkpoint with `New_baseline/infer.py` is still needed. For the new training checkpoint, keep the new architecture path and rely on its full `train_config.json`.
