#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

NS_TOKENIZER_TYPE="${NS_TOKENIZER_TYPE:-rankmixer}"
USER_NS_TOKENS="${USER_NS_TOKENS:-5}"
ITEM_NS_TOKENS="${ITEM_NS_TOKENS:-2}"
NUM_QUERIES="${NUM_QUERIES:-2}"
NS_GROUPS_JSON="${NS_GROUPS_JSON:-${SCRIPT_DIR}/ns_groups.json}"
EMB_SKIP_THRESHOLD="${EMB_SKIP_THRESHOLD:-1000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:128,seq_b:128,seq_c:256,seq_d:256}"
D_MODEL="${D_MODEL:-64}"
NUM_HYFORMER_BLOCKS="${NUM_HYFORMER_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-longer}"
SEQ_TOP_K="${SEQ_TOP_K:-50}"
REQUESTED_LOSS_TYPE="${LOSS_TYPE:-bce_pairwise}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_BATCHES="${BUFFER_BATCHES:-8}"
PAIRWISE_AUC_WEIGHT="${PAIRWISE_AUC_WEIGHT:-0.05}"
PAIRWISE_MAX_PAIRS="${PAIRWISE_MAX_PAIRS:-8192}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.1}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

LOSS_TYPE="${REQUESTED_LOSS_TYPE}"
PAIRWISE_SUPPORTED=0
if grep -q "bce_pairwise" "${SCRIPT_DIR}/train.py"; then
    PAIRWISE_SUPPORTED=1
fi
if [ "${LOSS_TYPE}" = "bce_pairwise" ] && [ "${PAIRWISE_SUPPORTED}" -ne 1 ]; then
    echo "PLATFORM_TRAIN_LOSS_FALLBACK {\"requested\":\"bce_pairwise\",\"fallback\":\"focal\",\"reason\":\"runtime_train_py_has_no_bce_pairwise\"}"
    LOSS_TYPE="focal"
fi

EXTRA_LOSS_ARGS=()
if [ "${LOSS_TYPE}" = "bce_pairwise" ]; then
    EXTRA_LOSS_ARGS+=(--pairwise_auc_weight "${PAIRWISE_AUC_WEIGHT}")
    EXTRA_LOSS_ARGS+=(--pairwise_max_pairs "${PAIRWISE_MAX_PAIRS}")
elif [ "${LOSS_TYPE}" = "focal" ]; then
    EXTRA_LOSS_ARGS+=(--focal_alpha "${FOCAL_ALPHA}")
    EXTRA_LOSS_ARGS+=(--focal_gamma "${FOCAL_GAMMA}")
fi

echo "PLATFORM_TRAIN_CONFIG {\"ns_tokenizer_type\":\"${NS_TOKENIZER_TYPE}\",\"user_ns_tokens\":${USER_NS_TOKENS},\"item_ns_tokens\":${ITEM_NS_TOKENS},\"num_queries\":${NUM_QUERIES},\"ns_groups_json\":\"${NS_GROUPS_JSON}\",\"emb_skip_threshold\":${EMB_SKIP_THRESHOLD},\"num_workers\":${NUM_WORKERS},\"seq_max_lens\":\"${SEQ_MAX_LENS}\",\"d_model\":${D_MODEL},\"num_hyformer_blocks\":${NUM_HYFORMER_BLOCKS},\"num_heads\":${NUM_HEADS},\"seq_encoder_type\":\"${SEQ_ENCODER_TYPE}\",\"seq_top_k\":${SEQ_TOP_K},\"requested_loss_type\":\"${REQUESTED_LOSS_TYPE}\",\"loss_type\":\"${LOSS_TYPE}\",\"pairwise_supported\":${PAIRWISE_SUPPORTED},\"batch_size\":${BATCH_SIZE},\"buffer_batches\":${BUFFER_BATCHES},\"pairwise_auc_weight\":${PAIRWISE_AUC_WEIGHT},\"pairwise_max_pairs\":${PAIRWISE_MAX_PAIRS},\"focal_alpha\":${FOCAL_ALPHA},\"focal_gamma\":${FOCAL_GAMMA}}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type "${NS_TOKENIZER_TYPE}" \
    --user_ns_tokens "${USER_NS_TOKENS}" \
    --item_ns_tokens "${ITEM_NS_TOKENS}" \
    --num_queries "${NUM_QUERIES}" \
    --ns_groups_json "${NS_GROUPS_JSON}" \
    --emb_skip_threshold "${EMB_SKIP_THRESHOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --buffer_batches "${BUFFER_BATCHES}" \
    --seq_max_lens "${SEQ_MAX_LENS}" \
    --d_model "${D_MODEL}" \
    --num_hyformer_blocks "${NUM_HYFORMER_BLOCKS}" \
    --num_heads "${NUM_HEADS}" \
    --seq_encoder_type "${SEQ_ENCODER_TYPE}" \
    --seq_top_k "${SEQ_TOP_K}" \
    --loss_type "${LOSS_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_LOSS_ARGS[@]}" \
    "$@"
