#!/bin/bash

# 训练阶段环境变量设置
export TRAIN_LOG_PATH="./logs"
export TRAIN_TF_EVENTS_PATH="./tf_events"
export TRAIN_DATA_PATH="./dataset/TencentGR_1k"
export TRAIN_CKPT_PATH="./ckpt"
export USER_CACHE_PATH="./user_cache"

export RUNTIME_SCRIPT_DIR="C:/Projects/Python/Tencent"

# 创建必要的目录
mkdir -p "${TRAIN_LOG_PATH}"
mkdir -p "${TRAIN_TF_EVENTS_PATH}"
mkdir -p "${TRAIN_CKPT_PATH}"
mkdir -p "${USER_CACHE_PATH}"

echo "Environment variables set:"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "TRAIN_CKPT_PATH: ${TRAIN_CKPT_PATH}"
echo "USER_CACHE_PATH: ${USER_CACHE_PATH}"

# 进入训练工作目录
cd ${RUNTIME_SCRIPT_DIR}

# 使用HSTU稀疏注意力和优化梯度裁剪进行训练
echo "Starting training with HSTU sparse attention and optimized gradient clipping..."
python -u main.py \
    --use_hstu \
    --hstu_local_window 64 \
    --hstu_use_global_token \
    --hstu_global_ratio 0.1 \
    --temp 0.12 \
    --adaptive_temp \
    --norm_output \
    --gradient_clip_strategy adaptive \
    --base_clip_norm 1.0 \
    --enable_modal_balancing \
    --gradient_history_window 100
