#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: improved RankMixer stack (no ns_groups.json required) ----
# Training-only recent-crop regularization keeps validation/inference full-history
# while reducing dependence on exact long-history coverage.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --high_cardinality_mode hash \
    --rank_mixer_mode learned \
    --use_wide_dense_tokens \
    --dense_scalar_transform signlog \
    --use_xdomain_features \
    --use_inter_time_buckets \
    --use_continuous_time \
    --use_target_aware_attn \
    --use_cross_domain_attn \
    --ffn_type swiglu \
    --norm_type rms \
    --warmup_steps 1000 \
    --min_lr_ratio 0.1 \
    --use_ema \
    --random_id_mask_prob 0.05 \
    --train_recent_crop_prob 0.18 \
    --train_recent_crop_min_ratio 0.60 \
    --num_workers 8 \
    --seed 20260515 \
    --log_every_n_steps 200 \
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
