#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="exp_20260614_short40_anomaly_vectors_v1"
CODE_ROOT="/home/expand_disk/code_repository/mfl/token_compression"
DATA_ROOT="/home/expand_disk/data_repository/mfl/token_compression/20260613_data"
PYTHON="/home/mfl/.conda/envs/token_compression/bin/python"
TRAIN_SCRIPT="${CODE_ROOT}/scripts/anomaly_vector/train_object_anomaly_vectors.py"
PRECACHE_SCRIPT="${CODE_ROOT}/scripts/anomaly_vector/precache_qwen_frame_tokens.py"
LOG_DIR="${DATA_ROOT}/results/${EXP_NAME}"
FRAME_TOKEN_CACHE_DIR="${DATA_ROOT}/results/qwen3vl_visual_token_cache_long1280"

mkdir -p "${LOG_DIR}"
cd "${CODE_ROOT}"

COMMON_ARGS=(
  --exp-name "${EXP_NAME}"
  --train-count 32
  --val-count 10
  --openset-count 3
  --max-frames 1600
  --window-frames 8
  --positives-per-event-object 1
  --neg-per-pos 1
  --epochs 80
  --batch-size 64
  --anomaly-vectors 8
  --threshold 0.5
  --run-token-evidence
  --token-pooling topk
  --token-topk-ratio 0.2
  --extract-batch-size 1
  --resize-long-edge 1280
  --frame-token-cache-dir "${FRAME_TOKEN_CACHE_DIR}"
)

echo "[start] $(date '+%F %T')"
echo "[exp] ${EXP_NAME}"
echo "[strategy] anomaly vectors only; low score means normal"
echo "[anomaly_vectors] 8"
echo "[cache] ${FRAME_TOKEN_CACHE_DIR}"

echo "[prepare] $(date '+%F %T')"
"${PYTHON}" "${TRAIN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --prepare-only \
  --device cpu \
  --train-device cpu \
  2>&1 | tee "${LOG_DIR}/prepare.log"

echo "[precache-start] $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${PRECACHE_SCRIPT}" \
  --exp-name "${EXP_NAME}" \
  --cache-dir "${FRAME_TOKEN_CACHE_DIR}" \
  --resize-long-edge 1280 \
  --device cuda:0 \
  --shard-index 0 \
  --shard-count 2 \
  2>&1 | tee "${LOG_DIR}/precache_shard0_gpu0.log" &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${PRECACHE_SCRIPT}" \
  --exp-name "${EXP_NAME}" \
  --cache-dir "${FRAME_TOKEN_CACHE_DIR}" \
  --resize-long-edge 1280 \
  --device cuda:0 \
  --shard-index 1 \
  --shard-count 2 \
  2>&1 | tee "${LOG_DIR}/precache_shard1_gpu1.log" &
pid1=$!

wait "${pid0}"
wait "${pid1}"
echo "[precache-done] $(date '+%F %T')"

echo "[train-start] $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${TRAIN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --device cuda:0 \
  --train-device cuda:0 \
  2>&1 | tee "${LOG_DIR}/tmux_train.log"

echo "[done] $(date '+%F %T')"
