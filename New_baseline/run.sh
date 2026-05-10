#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NS_TOKENIZER_TYPE="${NS_TOKENIZER_TYPE:-rankmixer}"
USER_NS_TOKENS="${USER_NS_TOKENS:-5}"
ITEM_NS_TOKENS="${ITEM_NS_TOKENS:-2}"
NUM_QUERIES="${NUM_QUERIES:-2}"
NS_GROUPS_JSON="${NS_GROUPS_JSON:-}"
EMB_SKIP_THRESHOLD="${EMB_SKIP_THRESHOLD:-1000000}"
HIGH_CARDINALITY_MODE="${HIGH_CARDINALITY_MODE:-zero}"
RANK_MIXER_MODE="${RANK_MIXER_MODE:-learned}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:128,seq_b:128,seq_c:256,seq_d:256}"
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-longer}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_BATCHES="${BUFFER_BATCHES:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
D_MODEL="${D_MODEL:-64}"
NUM_HYFORMER_BLOCKS="${NUM_HYFORMER_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
RANDOM_ID_MASK_PROB="${RANDOM_ID_MASK_PROB:-0.05}"
USE_EMA="${USE_EMA:-0}"

EMA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then
    EMA_ARGS+=(--use_ema)
fi

echo "PLATFORM_TRAIN_CONFIG {\"ns_tokenizer_type\":\"${NS_TOKENIZER_TYPE}\",\"user_ns_tokens\":${USER_NS_TOKENS},\"item_ns_tokens\":${ITEM_NS_TOKENS},\"num_queries\":${NUM_QUERIES},\"ns_groups_json\":\"${NS_GROUPS_JSON}\",\"emb_skip_threshold\":${EMB_SKIP_THRESHOLD},\"high_cardinality_mode\":\"${HIGH_CARDINALITY_MODE}\",\"rank_mixer_mode\":\"${RANK_MIXER_MODE}\",\"seq_max_lens\":\"${SEQ_MAX_LENS}\",\"seq_encoder_type\":\"${SEQ_ENCODER_TYPE}\",\"seq_top_k\":${SEQ_TOP_K},\"batch_size\":${BATCH_SIZE},\"buffer_batches\":${BUFFER_BATCHES},\"num_workers\":${NUM_WORKERS},\"d_model\":${D_MODEL},\"num_hyformer_blocks\":${NUM_HYFORMER_BLOCKS},\"num_heads\":${NUM_HEADS},\"use_ema\":${USE_EMA},\"random_id_mask_prob\":${RANDOM_ID_MASK_PROB}}"

# ---- Active config: OOM-safe RankMixer stack (no ns_groups.json required) ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type "${NS_TOKENIZER_TYPE}" \
    --user_ns_tokens "${USER_NS_TOKENS}" \
    --item_ns_tokens "${ITEM_NS_TOKENS}" \
    --num_queries "${NUM_QUERIES}" \
    --ns_groups_json "${NS_GROUPS_JSON}" \
    --emb_skip_threshold "${EMB_SKIP_THRESHOLD}" \
    --high_cardinality_mode "${HIGH_CARDINALITY_MODE}" \
    --rank_mixer_mode "${RANK_MIXER_MODE}" \
    --use_wide_dense_tokens \
    --dense_scalar_transform signlog \
    --use_xdomain_features \
    --use_inter_time_buckets \
    --use_continuous_time \
    --use_target_aware_attn \
    --use_cross_domain_attn \
    --ffn_type swiglu \
    --norm_type rms \
    --warmup_steps "${WARMUP_STEPS}" \
    --min_lr_ratio "${MIN_LR_RATIO}" \
    --random_id_mask_prob "${RANDOM_ID_MASK_PROB}" \
    --seq_max_lens "${SEQ_MAX_LENS}" \
    --seq_encoder_type "${SEQ_ENCODER_TYPE}" \
    --seq_top_k "${SEQ_TOP_K}" \
    --batch_size "${BATCH_SIZE}" \
    --buffer_batches "${BUFFER_BATCHES}" \
    --num_workers "${NUM_WORKERS}" \
    --d_model "${D_MODEL}" \
    --num_hyformer_blocks "${NUM_HYFORMER_BLOCKS}" \
    --num_heads "${NUM_HEADS}" \
    "${EMA_ARGS[@]}" \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
