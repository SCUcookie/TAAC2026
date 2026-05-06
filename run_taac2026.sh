#!/bin/bash
set -euo pipefail

export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-./logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-./tf_events}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./dataset/TAAC2026_train}"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-./ckpt}"
export USER_CACHE_PATH="${USER_CACHE_PATH:-./user_cache}"

mkdir -p "$TRAIN_LOG_PATH" "$TRAIN_TF_EVENTS_PATH" "$TRAIN_CKPT_PATH" "$USER_CACHE_PATH"

python -u main.py \
  --batch_size "${BATCH_SIZE:-4096}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --hidden_units "${HIDDEN_UNITS:-128}" \
  --num_blocks "${NUM_BLOCKS:-2}" \
  --num_heads "${NUM_HEADS:-4}" \
  --max_train_batches "${MAX_TRAIN_BATCHES:-6000}"
