#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

/home/lcwt/.conda/envs/token_pruner_merge/bin/python scripts/tracking/track_detections_strong_sort.py \
  --dataset-root datasets/sha_ave_nwp/nwpu_test \
  --detections-dir datasets/sha_ave_nwp/nwpu_test/object_detection/detections \
  --output-dir datasets/sha_ave_nwp/nwpu_test/tracking/default_schema
