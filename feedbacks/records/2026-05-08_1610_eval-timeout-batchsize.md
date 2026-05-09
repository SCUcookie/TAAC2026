# Eval Timeout Batchsize

## Date

2026-05-08 16:10

## Experiment

Original baseline model platform evaluation after fixing the `utils` import.

## Configuration

- Evaluation code: `baseline/evaluation/infer.py`
- Model: original baseline checkpoint
- Platform dataset rows in log: `310000`
- Logged eval batch size before fix: `256`

## Result

Inference started successfully but platform stopped it with `[PLATFORM] infer stage timed out`.

## Log Notes

- The log reached `PLATFORM_FIRST_BATCH`, so model loading and first forward pass worked.
- `PLATFORM_DATASET_CONFIG` showed `batch_size=256`, inherited from the training config.
- For `310000` rows, that requires about `1211` inference batches, which is too slow with long sequence lengths.

## Decision

Patch evaluation to use a larger inference batch by default and add progress logging.

## Next Action

Resubmit with patched `infer.py`. If timeout still happens, set `EVAL_BATCH_SIZE` higher or use `EVAL_SEQ_LEN_CAP` as an emergency speed-quality tradeoff.
