# TAAC2026 pCVR Submission V1

This repository now supports the 2026 flat pCVR/AUC task in a separate path while preserving the older TencentGR TopK code.

## What Runs

- `main.py` automatically dispatches to `taac2026_train.py` when `TRAIN_DATA_PATH` contains TAAC2026 Parquet files with columns like `user_id`, `item_id`, `user_int_feats_*`, and `domain_*_seq_*`.
- `infer.py` automatically dispatches to `taac2026_infer.py` when `EVAL_DATA_PATH` has the same flat schema.
- Older `seq.jsonl` / `predict_set.jsonl` TencentGR data still uses the existing retrieval implementation.

## Model

The first version is a compact single-model scorer:

- scalar categorical fields, multi-value fields, and sequence fields are hashed into per-field tokens;
- dense/list<float> fields are averaged into field tokens;
- time columns are encoded with hour/weekday sinusoidal features and label-time/timestamp gap;
- all tokens are processed by a small Transformer encoder, then an MLP outputs one pCVR logit.

This follows the competition direction of a unified block over non-sequential and sequential tokens, but keeps the model small enough for the official latency constraint.

## Training

```bash
export TRAIN_DATA_PATH=/path/to/taac2026/train
export TRAIN_CKPT_PATH=./ckpt
export TRAIN_LOG_PATH=./logs
bash run_taac2026.sh
```

Useful knobs:

- `MAX_TRAIN_BATCHES=6000`
- `BATCH_SIZE=4096`
- `HIDDEN_UNITS=128`
- `NUM_BLOCKS=2`
- `NUM_HEADS=4`

## Inference

```bash
export EVAL_DATA_PATH=/path/to/taac2026/eval
export MODEL_OUTPUT_PATH=./ckpt
export EVAL_RESULT_PATH=./eval_result
python infer.py
```

Outputs:

- `result.json`: JSON payload with `predictions`;
- `submission.csv`: `user_id,item_id,score`;
- `scores.npy`: raw float32 scores;
- `inference_summary.json`: count, checkpoint, and timing summary.

If the official platform expects a different filename or JSON key, change `--output_name` or `write_outputs()` in `taac2026_infer.py`.
