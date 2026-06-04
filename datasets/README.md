# Dataset Object Labels

This directory organizes dataset metadata and pseudo labels for object-centric token compression experiments.

## Current Collection

The current extensible collection entry point is:

- `sha_ave_nwp/`

It contains:

- `shanghaitech_test/`
- `avenue_test/`
- `nwpu_test/`
- `schemes/scheme1_dataset_specific/`
- `schemes/scheme2_unified_common/`

Future datasets should be added under `sha_ave_nwp/` and registered in `sha_ave_nwp/collection.json`.

## Layout

The older top-level dataset folders are still used as local data roots by running jobs:

- `shanghaitech_test/`
- `avenue_test/`
- `nwpu_test/`

Each dataset directory contains:

- `categories.txt`: object prompts chosen from the dataset anomaly definitions.
- `detections/`: one JSONL per test video, one record per frame.
- `work/`: multi-worker manifests and logs.
- `metadata.json`: source paths and anomaly/object rationale.
