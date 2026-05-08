# TAAC2026 pCVR Submission V1

This repository now supports the 2026 flat pCVR/AUC task in a separate path while preserving the older TencentGR TopK code.

## What Runs

- `main.py` automatically dispatches to `taac2026_train.py` when `TRAIN_DATA_PATH` contains TAAC2026 Parquet files with columns like `user_id`, `item_id`, `user_int_feats_*`, and `domain_*_seq_*`.
- `infer.py` automatically dispatches to `taac2026_infer.py` when `EVAL_DATA_PATH` has the same flat schema.
- Older `seq.jsonl` / `predict_set.jsonl` TencentGR data still uses the existing retrieval implementation.

## Model

The first version now follows the public description of the official 2026 baseline, PCVRHyFormer:

- `ns_groups.json`-style non-sequential feature groups are built from user, item, id, dense, and time fields;
- the four domain sequence families are encoded separately with `LongerEncoder` blocks;
- a learned multi-sequence query fuses domain summaries;
- HyFormer blocks jointly mix non-sequential tokens, sequence summaries, and time tokens;
- RankMixer-style interaction features feed the final pCVR head.

This keeps the official-baseline shape while using hashing so the first version can run without a full vocabulary-building pass.

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
- `SEQ_LEN=48`
- `VAL_STRATEGY=tail`
- `PAIRWISE_AUC_WEIGHT=0.05`
- `FOCAL_GAMMA=0.0`

The current candidate training recipe keeps the baseline architecture stable, increases the sequence window to 48, validates on a held-out tail window, and adds a small pairwise AUC-ranking loss on top of BCE.

## Local Labeled Evaluation

Use this before spending a platform evaluation slot when the eval split has labels:

```bash
export EVAL_DATA_PATH=/path/to/taac2026/labeled_eval
export MODEL_OUTPUT_PATH=./ckpt
export EVAL_RESULT_PATH=./eval_result
python taac2026_eval.py --experiment_name taac2026-tail-pairwise
```

Outputs:

- `labeled_result.json`: platform-shaped predictions for inspection.
- `labeled_eval_summary.json`: AUC, accuracy, logloss, threshold, and score distribution.
- `feedbacks/records/YYYY-MM-DD_HHMM_*.md`: feedback log for the run.

## Inference

```bash
export EVAL_DATA_PATH=/path/to/taac2026/eval
export MODEL_OUTPUT_PATH=./ckpt
export EVAL_RESULT_PATH=./eval_result
python infer.py
```

Outputs:

- `predictions.json`: platform-required JSON payload with `predictions`.
- `submission.csv`: optional local helper output when `--write_aux_outputs` is enabled.
- `scores.npy`: optional local helper output when `--write_aux_outputs` is enabled.
- `inference_summary.json`: optional local helper output when `--write_aux_outputs` is enabled.

Before spending an evaluation slot, run local inference once and validate the output shape with `submission/taac2026_baseline_v1/verify_infer.py`.
