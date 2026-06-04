#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

/home/lcwt/.conda/envs/token_pruner_merge/bin/python scripts/tracking/track_detections_strong_sort.py \
  --dataset-root datasets/avenue_test \
  --output-dir datasets/avenue_test/tracks_strong_sort
