#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="exp_20260614_event8_cover_short22_v1"
CODE_ROOT="/home/expand_disk/code_repository/mfl/token_compression"
DATA_ROOT="/home/expand_disk/data_repository/mfl/token_compression/20260613_data"
CACHE_DIR="/home/expand_disk/data_repository/mfl/token_compression/cache/20260613_data_token_cache"
PYTHON="/home/mfl/.conda/envs/token_compression/bin/python"
TRAIN_SCRIPT="${CODE_ROOT}/scripts/anomaly_vector/train_object_anomaly_vectors.py"
PRECACHE_SCRIPT="${CODE_ROOT}/scripts/anomaly_vector/precache_qwen_frame_tokens.py"
LOG_DIR="${DATA_ROOT}/results/${EXP_NAME}"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
cd "${CODE_ROOT}"

COMMON_ARGS=(
  --exp-name "${EXP_NAME}"
  --train-count 16
  --val-count 5
  --openset-count 1
  --max-frames 1600
  --window-frames 8
  --cover-event-windows
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
  --frame-token-cache-dir "${CACHE_DIR}"
)

echo "[start] $(date '+%F %T')"
echo "[exp] ${EXP_NAME}"
echo "[strategy] complete event coverage with 8-frame object windows"
echo "[normal negatives] strong=non-event time spans; medium=unrelated objects inside event span"
echo "[cache] ${CACHE_DIR}"

echo "[prepare] $(date '+%F %T')"
"${PYTHON}" "${TRAIN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --prepare-only \
  --device cpu \
  --train-device cpu \
  2>&1 | tee "${LOG_DIR}/prepare.log"

echo "[sample-stats] $(date '+%F %T')"
"${PYTHON}" - <<'PY' 2>&1 | tee "${LOG_DIR}/sample_frame_stats.log"
import json
from collections import Counter
from pathlib import Path

exp = Path("/home/expand_disk/data_repository/mfl/token_compression/20260613_data/results/exp_20260614_event8_cover_short22_v1")
cache = Path("/home/expand_disk/data_repository/mfl/token_compression/cache/20260613_data_token_cache")
frames = set()
labels = Counter()
neg = Counter()
for line in (exp / "sample_index.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    labels[(row["split"], row["label"], bool(row["is_positive"]))] += 1
    if not row["is_positive"]:
        neg[row.get("negative_strength") or "unknown"] += 1
    start, end = [int(x) for x in row["time_range"]]
    for frame_idx in range(start, end + 1):
        frames.add((row["package_id"], frame_idx))
missing = sum(1 for package_id, frame_idx in frames if not (cache / package_id / f"{frame_idx:06d}.pt").exists())
print(json.dumps({
    "sample_count_by_split_label_positive": {str(k): v for k, v in labels.items()},
    "negative_strength": dict(neg),
    "unique_frames": len(frames),
    "missing_cache_frames": missing,
}, ensure_ascii=False, indent=2))
PY

echo "[precache-start] $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${PRECACHE_SCRIPT}" \
  --exp-name "${EXP_NAME}" \
  --cache-dir "${CACHE_DIR}" \
  --resize-long-edge 1280 \
  --batch-size 1 \
  --device cuda:0 \
  --shard-index 0 \
  --shard-count 2 \
  2>&1 | tee "${LOG_DIR}/precache_shard0_gpu1.log" &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${PRECACHE_SCRIPT}" \
  --exp-name "${EXP_NAME}" \
  --cache-dir "${CACHE_DIR}" \
  --resize-long-edge 1280 \
  --batch-size 1 \
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
