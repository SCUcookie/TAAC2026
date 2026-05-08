#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SWEEP_NAME="${SWEEP_NAME:-seq_len_2026_05_08}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?TRAIN_DATA_PATH must point to the TAAC2026 training parquet directory}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/experiments/${SWEEP_NAME}}"

TRAIN_RATIO="${TRAIN_RATIO:-0.20}"
VALID_RATIO="${VALID_RATIO:-0.10}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BUFFER_BATCHES="${BUFFER_BATCHES:-8}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-1000}"

mkdir -p "${BASE_OUTPUT_DIR}"

run_candidate() {
    local name="$1"
    local seq_max_lens="$2"
    local run_dir="${BASE_OUTPUT_DIR}/${name}"

    mkdir -p "${run_dir}/ckpt" "${run_dir}/logs" "${run_dir}/tf_events"

    {
        echo "# ${name}"
        echo "- started_at: $(date -Iseconds)"
        echo "- train_data_path: ${TRAIN_DATA_PATH}"
        echo "- train_ratio: ${TRAIN_RATIO}"
        echo "- valid_ratio: ${VALID_RATIO}"
        echo "- num_epochs: ${NUM_EPOCHS}"
        echo "- batch_size: ${BATCH_SIZE}"
        echo "- num_workers: ${NUM_WORKERS}"
        echo "- buffer_batches: ${BUFFER_BATCHES}"
        echo "- eval_every_n_steps: ${EVAL_EVERY_N_STEPS}"
        echo "- seq_max_lens: ${seq_max_lens}"
        echo
        echo "## Command"
        echo "\`\`\`bash"
        echo "TRAIN_CKPT_PATH=${run_dir}/ckpt TRAIN_LOG_PATH=${run_dir}/logs TRAIN_TF_EVENTS_PATH=${run_dir}/tf_events bash ${SCRIPT_DIR}/run.sh --seq_max_lens ${seq_max_lens} --train_ratio ${TRAIN_RATIO} --valid_ratio ${VALID_RATIO} --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --buffer_batches ${BUFFER_BATCHES} --eval_every_n_steps ${EVAL_EVERY_N_STEPS}"
        echo "\`\`\`"
        echo
    } > "${run_dir}/run_record.md"

    TRAIN_CKPT_PATH="${run_dir}/ckpt" \
    TRAIN_LOG_PATH="${run_dir}/logs" \
    TRAIN_TF_EVENTS_PATH="${run_dir}/tf_events" \
    bash "${SCRIPT_DIR}/run.sh" \
        --seq_max_lens "${seq_max_lens}" \
        --train_ratio "${TRAIN_RATIO}" \
        --valid_ratio "${VALID_RATIO}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --buffer_batches "${BUFFER_BATCHES}" \
        --eval_every_n_steps "${EVAL_EVERY_N_STEPS}" \
        2>&1 | tee "${run_dir}/console.log"

    {
        echo
        echo "## Completion"
        echo "- finished_at: $(date -Iseconds)"
        echo "- log_path: ${run_dir}/logs/train.log"
        echo "- checkpoint_dir: ${run_dir}/ckpt"
    } >> "${run_dir}/run_record.md"
}

run_candidate "baseline_a256" "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
run_candidate "candidate_a512" "seq_a:512,seq_b:256,seq_c:512,seq_d:512"
run_candidate "candidate_a768" "seq_a:768,seq_b:256,seq_c:512,seq_d:512"

cat <<EOF
Sweep complete.

Compare validation AUC/logloss and runtime across:
  ${BASE_OUTPUT_DIR}/baseline_a256
  ${BASE_OUTPUT_DIR}/candidate_a512
  ${BASE_OUTPUT_DIR}/candidate_a768

Do not use a platform evaluation slot until the selected checkpoint passes local inference verification.
EOF
