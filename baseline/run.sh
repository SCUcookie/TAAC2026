#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

NS_TOKENIZER_TYPE="${NS_TOKENIZER_TYPE:-rankmixer}"
USER_NS_TOKENS="${USER_NS_TOKENS:-5}"
ITEM_NS_TOKENS="${ITEM_NS_TOKENS:-2}"
NUM_QUERIES="${NUM_QUERIES:-2}"
NS_GROUPS_JSON="${NS_GROUPS_JSON:-}"
EMB_SKIP_THRESHOLD="${EMB_SKIP_THRESHOLD:-1000000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEQ_MAX_LENS="${SEQ_MAX_LENS:-seq_a:256,seq_b:256,seq_c:512,seq_d:512}"
D_MODEL="${D_MODEL:-64}"
NUM_HYFORMER_BLOCKS="${NUM_HYFORMER_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
SEQ_ENCODER_TYPE="${SEQ_ENCODER_TYPE:-transformer}"
LOSS_TYPE="${LOSS_TYPE:-bce}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PAIRWISE_AUC_WEIGHT="${PAIRWISE_AUC_WEIGHT:-0.05}"
PAIRWISE_MAX_PAIRS="${PAIRWISE_MAX_PAIRS:-8192}"

echo "PLATFORM_TRAIN_CONFIG {\"ns_tokenizer_type\":\"${NS_TOKENIZER_TYPE}\",\"user_ns_tokens\":${USER_NS_TOKENS},\"item_ns_tokens\":${ITEM_NS_TOKENS},\"num_queries\":${NUM_QUERIES},\"ns_groups_json\":\"${NS_GROUPS_JSON}\",\"emb_skip_threshold\":${EMB_SKIP_THRESHOLD},\"num_workers\":${NUM_WORKERS},\"seq_max_lens\":\"${SEQ_MAX_LENS}\",\"d_model\":${D_MODEL},\"num_hyformer_blocks\":${NUM_HYFORMER_BLOCKS},\"num_heads\":${NUM_HEADS},\"seq_encoder_type\":\"${SEQ_ENCODER_TYPE}\",\"loss_type\":\"${LOSS_TYPE}\",\"batch_size\":${BATCH_SIZE},\"pairwise_auc_weight\":${PAIRWISE_AUC_WEIGHT},\"pairwise_max_pairs\":${PAIRWISE_MAX_PAIRS}}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type "${NS_TOKENIZER_TYPE}" \
    --user_ns_tokens "${USER_NS_TOKENS}" \
    --item_ns_tokens "${ITEM_NS_TOKENS}" \
    --num_queries "${NUM_QUERIES}" \
    --ns_groups_json "${NS_GROUPS_JSON}" \
    --emb_skip_threshold "${EMB_SKIP_THRESHOLD}" \
    --num_workers "${NUM_WORKERS}" \
    --seq_max_lens "${SEQ_MAX_LENS}" \
    --d_model "${D_MODEL}" \
    --num_hyformer_blocks "${NUM_HYFORMER_BLOCKS}" \
    --num_heads "${NUM_HEADS}" \
    --seq_encoder_type "${SEQ_ENCODER_TYPE}" \
    --loss_type "${LOSS_TYPE}" \
    --pairwise_auc_weight "${PAIRWISE_AUC_WEIGHT}" \
    --pairwise_max_pairs "${PAIRWISE_MAX_PAIRS}" \
    --batch_size "${BATCH_SIZE}" \
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
