#!/bin/bash

export TRAIN_LOG_PATH="./logs"
export TRAIN_TF_EVENTS_PATH="./tf_events"
export TRAIN_DATA_PATH="./dataset/TencentGR_1k"
export TRAIN_CKPT_PATH="./ckpt"

export RUNTIME_SCRIPT_DIR="C:/Projects/Python/Tencent"

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py
