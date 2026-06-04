# Dataset Workspace

This directory organizes dataset payloads, metadata, object detections, tracking outputs, and anomaly schemes for object-centric token compression experiments.

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

## Active Policy

`sha_ave_nwp/` is the only active local dataset workspace. Concrete dataset payloads should be copied or moved into the relevant dataset folder under this collection so the project does not depend on external source paths.

Large payloads remain ignored by Git, but they should exist locally inside `sha_ave_nwp/` when a dataset is ready for experiments.

Old top-level dataset entries are being removed after their payloads are copied into `sha_ave_nwp/`.

Current status:

- `shanghaitech_test`: materialized under `sha_ave_nwp/shanghaitech_test`; old top-level entry removed.
- `nwpu_test`: materialized under `sha_ave_nwp/nwpu_test`; old top-level entry removed.
- `avenue_test`: frames are materialized under `sha_ave_nwp/avenue_test`; detection repair is in progress for missing frames.

## Dataset Folder Contents

Each dataset folder may contain:

- `frames/`: concrete test frames.
- `videos/`: raw test videos when available or needed.
- `detections/` or detector-specific detection folders: one JSONL per test video, one record per frame.
- `object_detection/`: stable internal links to the active detection results and categories.
- `tracking/`: per-scheme tracking outputs.
- `anomaly_types/`: dataset-local copies of supported scheme definitions.
- `metadata/`: object categories, annotation design, and legacy metadata.
- `frames_GT.bak/`: old frame-level labels kept only as backup, not active supervision.
